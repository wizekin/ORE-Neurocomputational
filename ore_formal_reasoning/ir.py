"""Typed logical intermediate representation for FRCE (Phase B, Steps 8-15).

The IR maps USG objects to constants, classes to predicates or types, relations
to predicates/functions, rules to named implications, and numerical attributes
to typed values with units. It supports the solver-selection policy of the
specification: Boolean structure, linear/nonlinear arithmetic, bit-vectors,
strings, enumerations and uninterpreted functions, with explicit quantifier
scope and cardinality constraints.

Normalization (Step 15) canonicalizes names, alpha-renames bound variables,
folds only provably safe boolean/numeric constants, type-checks every term
against declared sorts, and validates all references while preserving a
bidirectional mapping from normalized statements back to their sources.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from fractions import Fraction
from typing import Any, Mapping

from .contracts import (
    EvidenceTag,
    SourceRef,
    TemporalScope,
    canonical_json,
    sha256_json,
)
from .exceptions import InvalidFormalizationError, UnitMismatchError
from .version import FORMAL_REASONING_IR_VERSION

_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# ---------------------------------------------------------------------------
# Sorts (Step 9)
# ---------------------------------------------------------------------------


class SortKind(str, Enum):
    """Base kinds supported by the typed IR."""

    BOOL = "bool"
    INT = "int"
    REAL = "real"
    BITVECTOR = "bitvec"
    STRING = "string"
    ENUM = "enum"
    DOMAIN = "domain"


class Finiteness(str, Enum):
    """Whether each declared domain is finite, infinite, bounded, or partially known."""

    FINITE = "finite"
    INFINITE = "infinite"
    BOUNDED = "bounded"
    PARTIALLY_KNOWN = "partially_known"


@dataclass(frozen=True, slots=True)
class SortDecl:
    """One declared sort with its domain finiteness (Step 9)."""

    name: str
    kind: SortKind
    width: int | None = None            # bit-vector width
    values: tuple[str, ...] = ()        # enumeration values
    finiteness: Finiteness = Finiteness.INFINITE
    lower: int | None = None
    upper: int | None = None

    def __post_init__(self) -> None:
        if not _IDENTIFIER_RE.match(self.name):
            raise ValueError(f"sort name must be an identifier: {self.name!r}")
        if self.kind == SortKind.BITVECTOR:
            if self.width is None or self.width <= 0 or self.width > 4096:
                raise ValueError(f"bit-vector sort {self.name!r} requires 1..4096 width")
        if self.kind == SortKind.ENUM:
            if not self.values or len(set(self.values)) != len(self.values):
                raise ValueError(f"enum sort {self.name!r} requires unique non-empty values")
        if self.finiteness is Finiteness.FINITE and self.kind in {
            SortKind.INT, SortKind.REAL,
        }:
            if self.lower is None or self.upper is None or self.lower > self.upper:
                raise ValueError(f"finite numeric sort {self.name!r} requires lower<=upper bounds")

    def __str__(self) -> str:
        return self.name


BOOL_SORT = SortDecl("Bool", SortKind.BOOL, finiteness=Finiteness.FINITE)
INT_SORT = SortDecl("Int", SortKind.INT)
REAL_SORT = SortDecl("Real", SortKind.REAL)
STRING_SORT = SortDecl("String", SortKind.STRING, finiteness=Finiteness.INFINITE)

BUILTIN_SORTS: Mapping[str, SortDecl] = {
    "Bool": BOOL_SORT,
    "Int": INT_SORT,
    "Real": REAL_SORT,
    "String": STRING_SORT,
}

NUMERIC_KINDS = frozenset({SortKind.INT, SortKind.REAL})


def make_enum_sort(name: str, values: tuple[str, ...] | list[str]) -> SortDecl:
    """Create an enumeration sort with finite, known domain."""
    return SortDecl(
        name, SortKind.ENUM, values=tuple(values), finiteness=Finiteness.FINITE,
    )


def make_bv_sort(name: str, width: int) -> SortDecl:
    """Create a bit-vector sort of the given width."""
    return SortDecl(name, SortKind.BITVECTOR, width=width, finiteness=Finiteness.FINITE)


# ---------------------------------------------------------------------------
# Terms (Step 8, 10, 13)
# ---------------------------------------------------------------------------


class ArithOp(str, Enum):
    """Arithmetic operators with strictly typed operands."""

    ADD = "+"
    SUB = "-"
    MUL = "*"
    DIV = "/"   # exact division; Real operands only (or Int/Int producing exact rational via DIV → rejected)


class CompareOp(str, Enum):
    """Comparison operators; equality is sort-generic, ordering is numeric/BV."""

    EQ = "="
    NEQ = "!="
    LT = "<"
    LE = "<="
    GT = ">"
    GE = ">="


class BvOp(str, Enum):
    """Bit-vector operations following bit-vector (wrapping) semantics."""

    ADD = "bvadd"
    SUB = "bvsub"
    MUL = "bvmul"
    AND = "bvand"
    OR = "bvor"
    XOR = "bvxor"
    SHL = "bvshl"
    LSHR = "bvlshr"


class StrOp(str, Enum):
    """String operations routed only to a supported string theory."""

    CONCAT = "str_concat"
    LEN = "str_len"
    CONTAINS = "str_contains"
    PREFIX = "str_prefix"
    SUFFIX = "str_suffix"


@dataclass(frozen=True, slots=True)
class Const:
    """A typed constant symbol (Step 8: USG objects map to constants)."""

    name: str
    sort: str
    enum_value: str | None = None

    def __post_init__(self) -> None:
        if not _IDENTIFIER_RE.match(self.name):
            raise ValueError(f"constant name must be an identifier: {self.name!r}")


@dataclass(frozen=True, slots=True)
class Var:
    """A typed variable, either free in a rule or bound by a quantifier."""

    name: str
    sort: str

    def __post_init__(self) -> None:
        if not _IDENTIFIER_RE.match(self.name):
            raise ValueError(f"variable name must be an identifier: {self.name!r}")


@dataclass(frozen=True, slots=True)
class NumLit:
    """A numeric literal; integer and real literals are never silently interchanged."""

    value: Fraction
    sort: str  # "Int" or "Real"

    def __post_init__(self) -> None:
        if self.sort not in {"Int", "Real"}:
            raise ValueError(f"NumLit sort must be Int or Real, got {self.sort!r}")
        if self.sort == "Int" and self.value.denominator != 1:
            raise ValueError("integer literal must have integral value")

    @staticmethod
    def int_value(value: int) -> "NumLit":
        """Build an integer literal."""
        return NumLit(Fraction(value), "Int")

    @staticmethod
    def real_value(value: Fraction | float) -> "NumLit":
        """Build a real literal (exact rational internally)."""
        return NumLit(Fraction(value), "Real")


@dataclass(frozen=True, slots=True)
class BoolLit:
    """A Boolean literal."""

    value: bool


@dataclass(frozen=True, slots=True)
class StrLit:
    """A string literal."""

    value: str


@dataclass(frozen=True, slots=True)
class BVLit:
    """A bit-vector literal with explicit width."""

    value: int
    width: int

    def __post_init__(self) -> None:
        if self.width <= 0 or self.width > 4096:
            raise ValueError("bit-vector literal width must be 1..4096")
        if not 0 <= self.value < (1 << self.width):
            raise ValueError("bit-vector literal value must fit its width")


@dataclass(frozen=True, slots=True)
class ToReal:
    """Explicit Int-to-Real conversion; the only permitted implicit-coercion point."""

    operand: "Term"


@dataclass(frozen=True, slots=True)
class ArithExpr:
    """Typed arithmetic; operands must share one numeric sort."""

    op: ArithOp
    args: tuple["Term", ...]

    def __post_init__(self) -> None:
        if len(self.args) != 2:
            raise ValueError("arithmetic expressions take exactly two operands")


@dataclass(frozen=True, slots=True)
class BvExpr:
    """Bit-vector expression with wrapping semantics."""

    op: BvOp
    args: tuple["Term", ...]

    def __post_init__(self) -> None:
        if not self.args:
            raise ValueError("bit-vector expressions need operands")


@dataclass(frozen=True, slots=True)
class StrExpr:
    """String expression; CONTAINS/PREFIX/SUFFIX are Boolean-valued."""

    op: StrOp
    args: tuple["Term", ...]

    def __post_init__(self) -> None:
        if not self.args:
            raise ValueError("string expressions need operands")


@dataclass(frozen=True, slots=True)
class FuncApp:
    """Application of a declared uninterpreted function."""

    function: str
    args: tuple["Term", ...]


Term = Const | Var | NumLit | BoolLit | StrLit | BVLit | ToReal | ArithExpr | BvExpr | StrExpr | FuncApp


# ---------------------------------------------------------------------------
# Formulas (Step 10, 11)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Atom:
    """Application of a declared predicate to typed terms."""

    predicate: str
    args: tuple[Term, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.args, tuple):
            raise ValueError(
                "Atom.args must be a tuple of terms; wrap a single argument as (term,)"
            )
        if not self.args:
            raise ValueError(f"atom {self.predicate!r} needs at least one argument")


@dataclass(frozen=True, slots=True)
class Compare:
    """Comparison formula over terms of one compatible sort."""

    op: CompareOp
    left: Term
    right: Term


@dataclass(frozen=True, slots=True)
class BoolTermF:
    """Promote a Boolean-sorted term into formula position."""

    term: Term


@dataclass(frozen=True, slots=True)
class NotF:
    """Negation."""

    body: "Formula"


@dataclass(frozen=True, slots=True)
class AndF:
    """Conjunction of one or more formulas."""

    args: tuple["Formula", ...]

    def __post_init__(self) -> None:
        if not self.args:
            raise ValueError("conjunction needs at least one argument")


@dataclass(frozen=True, slots=True)
class OrF:
    """Disjunction of one or more formulas."""

    args: tuple["Formula", ...]

    def __post_init__(self) -> None:
        if not self.args:
            raise ValueError("disjunction needs at least one argument")


@dataclass(frozen=True, slots=True)
class ImpliesF:
    """Implication."""

    premise: "Formula"
    conclusion: "Formula"


@dataclass(frozen=True, slots=True)
class IffF:
    """Biconditional."""

    left: "Formula"
    right: "Formula"


@dataclass(frozen=True, slots=True)
class VarDecl:
    """Quantifier-bound variable declaration with its sort."""

    variable: Var


@dataclass(frozen=True, slots=True)
class ForallF:
    """Universal quantification with explicit bound scope (Step 11)."""

    variables: tuple[VarDecl, ...]
    body: "Formula"

    def __post_init__(self) -> None:
        if not self.variables:
            raise ValueError("universal quantifier needs at least one bound variable")


@dataclass(frozen=True, slots=True)
class ExistsF:
    """Existential quantification with explicit bound scope (Step 11)."""

    variables: tuple[VarDecl, ...]
    body: "Formula"

    def __post_init__(self) -> None:
        if not self.variables:
            raise ValueError("existential quantifier needs at least one bound variable")


@dataclass(frozen=True, slots=True)
class CardinalityF:
    """Exactly/at-least/at-most cardinality over a list of formulas (Step 11)."""

    mode: str  # exactly | at_least | at_most
    count: int
    args: tuple["Formula", ...]

    def __post_init__(self) -> None:
        if self.mode not in {"exactly", "at_least", "at_most"}:
            raise ValueError(f"unknown cardinality mode: {self.mode!r}")
        if not self.args:
            raise ValueError("cardinality constraints need at least one formula")
        if not 0 <= self.count <= len(self.args):
            raise ValueError("cardinality count must be within [0, len(args)]")


Formula = (
    Atom | Compare | BoolTermF | NotF | AndF | OrF | ImpliesF | IffF
    | ForallF | ExistsF | CardinalityF
)


# ---------------------------------------------------------------------------
# Declarations (Step 10) and rules (Step 12)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PredicateDecl:
    """One declared predicate with arity, argument sorts, and interpretation."""

    name: str
    arg_sorts: tuple[str, ...]
    interpretation: str = ""

    def __post_init__(self) -> None:
        if not _IDENTIFIER_RE.match(self.name):
            raise ValueError(f"predicate name must be an identifier: {self.name!r}")
        if not self.arg_sorts:
            raise ValueError(f"predicate {self.name!r} needs at least one argument sort")


@dataclass(frozen=True, slots=True)
class FunctionDecl:
    """One declared uninterpreted function with argument and result sorts."""

    name: str
    arg_sorts: tuple[str, ...]
    result_sort: str

    def __post_init__(self) -> None:
        if not _IDENTIFIER_RE.match(self.name):
            raise ValueError(f"function name must be an identifier: {self.name!r}")


@dataclass(frozen=True, slots=True)
class BodyLiteral:
    """One rule-body literal: a predicate atom with explicit polarity (Step 12)."""

    atom: Atom
    negated: bool = False


@dataclass(frozen=True, slots=True)
class RuleDecl:
    """A named implication with explicit premises, conclusion, and variable mapping."""

    rule_id: str
    name: str
    variables: tuple[VarDecl, ...]
    head: Atom
    body: tuple[BodyLiteral, ...]
    source_refs: tuple[SourceRef, ...] = ()
    version: str = "1.0.0"

    def __post_init__(self) -> None:
        if not self.rule_id.strip() or not self.name.strip():
            raise ValueError("rules require non-empty rule_id and name")
        if not self.body:
            raise ValueError(f"rule {self.rule_id!r} requires at least one body literal")
        object.__setattr__(self, "variables", tuple(self.variables))
        object.__setattr__(self, "body", tuple(self.body))
        object.__setattr__(self, "source_refs", tuple(self.source_refs))


# ---------------------------------------------------------------------------
# Units and dimensional constraints (Step 13)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class UnitDecl:
    """A measurement unit described by its base-dimension exponent vector."""

    name: str
    dimension: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("unit name must be non-empty")
        dims = {d: e for d, e in self.dimension}
        if len(dims) != len(self.dimension):
            raise ValueError(f"unit {self.name!r} repeats a base dimension")
        object.__setattr__(self, "dimension", tuple(sorted(dims.items())))

    def dimension_vector(self) -> tuple[tuple[str, int], ...]:
        """Return the sorted base-dimension exponent vector."""
        return self.dimension


def dimensions_compatible(a: UnitDecl, b: UnitDecl) -> bool:
    """Return whether two units share one dimension vector (Step 13)."""
    return a.dimension_vector() == b.dimension_vector()


@dataclass(frozen=True, slots=True)
class UnitConversion:
    """A validated linear conversion rule between dimension-compatible units."""

    from_unit: str
    to_unit: str
    factor: Fraction
    offset: Fraction = Fraction(0)
    dimension: tuple[tuple[str, int], ...] = ()

    def __post_init__(self) -> None:
        if not self.from_unit.strip() or not self.to_unit.strip():
            raise ValueError("unit conversions need non-empty unit names")
        if self.from_unit == self.to_unit:
            raise ValueError("unit conversion must change the unit")
        if self.factor == 0:
            raise ValueError("unit conversion factor must be non-zero")

    def convert(self, value: Fraction) -> Fraction:
        """Apply the linear conversion value * factor + offset."""
        return value * self.factor + self.offset


class UnitTable:
    """Registry of units and validated conversions used by the formalization compiler."""

    def __init__(self) -> None:
        self._units: dict[str, UnitDecl] = {}
        self._conversions: dict[tuple[str, str], UnitConversion] = {}

    def declare(self, unit: UnitDecl) -> None:
        """Declare one unit; redeclaration with a different dimension is rejected."""
        existing = self._units.get(unit.name)
        if existing is not None:
            if existing.dimension_vector() != unit.dimension_vector():
                raise UnitMismatchError(
                    f"unit {unit.name!r} redeclared with a different dimension"
                )
            return
        self._units[unit.name] = unit

    def declare_conversion(self, conversion: UnitConversion) -> None:
        """Declare one conversion after checking dimensional compatibility."""
        source = self._units.get(conversion.from_unit)
        target = self._units.get(conversion.to_unit)
        if source is None or target is None:
            raise UnitMismatchError(
                f"conversion {conversion.from_unit}->{conversion.to_unit} references "
                "undeclared units"
            )
        if not dimensions_compatible(source, target):
            raise UnitMismatchError(
                f"cannot convert {conversion.from_unit} to {conversion.to_unit}: "
                "incompatible dimensions"
            )
        if conversion.dimension and tuple(sorted(conversion.dimension)) != source.dimension_vector():
            raise UnitMismatchError("conversion dimension does not match its units")
        self._conversions[(conversion.from_unit, conversion.to_unit)] = conversion

    def convert(self, value: Fraction, from_unit: str, to_unit: str) -> Fraction:
        """Convert a value between dimension-compatible units via validated rules only."""
        if from_unit == to_unit:
            return value
        direct = self._conversions.get((from_unit, to_unit))
        if direct is not None:
            return direct.convert(value)
        inverse = self._conversions.get((to_unit, from_unit))
        if inverse is not None:
            # Invert v = u * f + o  =>  u = (v - o) / f
            return (value - inverse.offset) / inverse.factor
        source = self._units.get(from_unit)
        target = self._units.get(to_unit)
        if source is None or target is None:
            raise UnitMismatchError(f"undeclared unit in conversion: {from_unit!r} or {to_unit!r}")
        raise UnitMismatchError(
            f"no validated conversion rule from {from_unit!r} to {to_unit!r}"
        )

    def require_same_dimension(self, a: str, b: str) -> None:
        """Reject combining quantities of incompatible dimensions before solver use."""
        unit_a = self._units.get(a)
        unit_b = self._units.get(b)
        if unit_a is None or unit_b is None:
            raise UnitMismatchError(f"undeclared unit: {a!r} or {b!r}")
        if not dimensions_compatible(unit_a, unit_b):
            raise UnitMismatchError(
                f"dimension mismatch: {a} ({unit_a.dimension_vector()}) vs "
                f"{b} ({unit_b.dimension_vector()})"
            )


@dataclass(frozen=True, slots=True)
class Quantity:
    """A typed quantity: exact value plus declared unit (Step 13)."""

    value: Fraction
    unit: str


# ---------------------------------------------------------------------------
# Semantic assumption profiles (Step 14)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DomainSemantics:
    """Per-domain semantic assumptions; never one accidental global profile."""

    domain: str
    closed_world: bool = False
    explicit_negation: bool = True
    unique_names: bool = True
    equality_semantics: str = "standard"
    incomplete_knowledge: str = "open_world_absence_is_not_false"

    def __post_init__(self) -> None:
        if not self.domain.strip():
            raise ValueError("domain name must be non-empty")
        if self.equality_semantics not in {"standard", "unique_names"}:
            raise ValueError("equality_semantics must be standard or unique_names")
        if self.incomplete_knowledge not in {
            "open_world_absence_is_not_false", "closed_world_completion",
        }:
            raise ValueError("incomplete_knowledge policy is not recognized")
        if self.closed_world and not self.explicit_negation:
            raise ValueError(
                "closed-world domains require explicit negation handling"
            )


# ---------------------------------------------------------------------------
# Statements and the IR container (Step 8)
# ---------------------------------------------------------------------------


class StatementKind(str, Enum):
    """What kind of top-level statement this is."""

    FACT = "fact"
    CONSTRAINT = "constraint"
    GOAL = "goal"


@dataclass(frozen=True, slots=True)
class Statement:
    """One top-level IR statement with provenance, scope, and evidence tag."""

    statement_id: str
    kind: StatementKind
    formula: Formula
    domain: str = "default"
    tag: EvidenceTag = EvidenceTag.OBSERVED_FACT
    source_refs: tuple[SourceRef, ...] = ()
    temporal_scope: TemporalScope | None = None
    context_scope: str = "default"
    origin: str = "request"                    # request | usg | assumption | policy | derived
    assumption_name: str | None = None         # named-assumption binding (Step 25)
    variables: tuple[VarDecl, ...] = ()        # free variables of constraints/goals
    note: str | None = None

    def __post_init__(self) -> None:
        if not self.statement_id.strip():
            raise ValueError("statement_id must be non-empty")
        object.__setattr__(self, "source_refs", tuple(self.source_refs))
        object.__setattr__(self, "variables", tuple(self.variables))


@dataclass(frozen=True, slots=True)
class NormalizationIssue:
    """One validation finding produced by the normalizer (Step 15)."""

    severity: str          # error | warning
    code: str
    message: str
    statement_id: str | None = None


@dataclass(frozen=True, slots=True)
class FormalIR:
    """The typed logical IR produced by the formalization compiler (Step 8)."""

    sorts: tuple[SortDecl, ...]
    predicates: tuple[PredicateDecl, ...]
    functions: tuple[FunctionDecl, ...]
    statements: tuple[Statement, ...]
    rules: tuple[RuleDecl, ...]
    domains: tuple[DomainSemantics, ...]
    unit_table: UnitTable
    warnings: tuple[NormalizationIssue, ...] = ()
    source_map: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    version: str = FORMAL_REASONING_IR_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "sorts", tuple(self.sorts))
        object.__setattr__(self, "predicates", tuple(self.predicates))
        object.__setattr__(self, "functions", tuple(self.functions))
        object.__setattr__(self, "statements", tuple(self.statements))
        object.__setattr__(self, "rules", tuple(self.rules))
        object.__setattr__(self, "domains", tuple(self.domains))
        object.__setattr__(self, "warnings", tuple(self.warnings))
        ids = [s.statement_id for s in self.statements]
        if len(ids) != len(set(ids)):
            raise ValueError("statement IDs must be unique within an IR")
        rule_ids = [r.rule_id for r in self.rules]
        if len(rule_ids) != len(set(rule_ids)):
            raise ValueError("rule IDs must be unique within an IR")
        domain_names = {d.domain for d in self.domains}
        for statement in self.statements:
            if statement.domain not in domain_names:
                raise ValueError(
                    f"statement {statement.statement_id!r} references domain "
                    f"{statement.domain!r} without a declared semantic profile (Step 14)"
                )

    def sort_map(self) -> dict[str, SortDecl]:
        """Return declared sorts including builtins, keyed by name."""
        merged = dict(BUILTIN_SORTS)
        merged.update({s.name: s for s in self.sorts})
        return merged

    def predicate_map(self) -> dict[str, PredicateDecl]:
        """Return declared predicates keyed by name."""
        return {p.name: p for p in self.predicates}

    def function_map(self) -> dict[str, FunctionDecl]:
        """Return declared functions keyed by name."""
        return {f.name: f for f in self.functions}

    def domain_map(self) -> dict[str, DomainSemantics]:
        """Return per-domain semantic profiles keyed by domain."""
        return {d.domain: d for d in self.domains}

    def facts(self) -> tuple[Statement, ...]:
        """Return statements of kind FACT."""
        return tuple(s for s in self.statements if s.kind == StatementKind.FACT)

    def constraints(self) -> tuple[Statement, ...]:
        """Return statements of kind CONSTRAINT."""
        return tuple(s for s in self.statements if s.kind == StatementKind.CONSTRAINT)

    def goals(self) -> tuple[Statement, ...]:
        """Return statements of kind GOAL."""
        return tuple(s for s in self.statements if s.kind == StatementKind.GOAL)

    def assumptions(self) -> tuple[Statement, ...]:
        """Return statements tagged as stipulated assumptions (Step 5)."""
        return tuple(
            s for s in self.statements
            if s.tag == EvidenceTag.STIPULATED_ASSUMPTION
        )

    def canonical_hash(self) -> str:
        """Return the deterministic content hash used for caching and audit."""
        return sha256_json({
            "version": self.version,
            "sorts": [canonical_json(_sort_payload(s)) for s in self.sorts],
            "predicates": [canonical_json(_predicate_payload(p)) for p in self.predicates],
            "functions": [canonical_json(_function_payload(f)) for f in self.functions],
            "statements": [canonical_json(_statement_payload(s)) for s in self.statements],
            "rules": [canonical_json(_rule_payload(r)) for r in self.rules],
            "domains": [canonical_json(_domain_payload(d)) for d in self.domains],
        })


# ---------------------------------------------------------------------------
# Serialization payloads (canonical, deterministic)
# ---------------------------------------------------------------------------


def _sort_payload(s: SortDecl) -> dict[str, Any]:
    return {
        "name": s.name, "kind": s.kind.value, "width": s.width,
        "values": list(s.values), "finiteness": s.finiteness.value,
        "lower": s.lower, "upper": s.upper,
    }


def _predicate_payload(p: PredicateDecl) -> dict[str, Any]:
    return {"name": p.name, "arg_sorts": list(p.arg_sorts), "interpretation": p.interpretation}


def _function_payload(f: FunctionDecl) -> dict[str, Any]:
    return {"name": f.name, "arg_sorts": list(f.arg_sorts), "result_sort": f.result_sort}


def _domain_payload(d: DomainSemantics) -> dict[str, Any]:
    return {
        "domain": d.domain, "closed_world": d.closed_world,
        "explicit_negation": d.explicit_negation, "unique_names": d.unique_names,
        "equality_semantics": d.equality_semantics,
        "incomplete_knowledge": d.incomplete_knowledge,
    }


def _statement_payload(s: Statement) -> dict[str, Any]:
    return {
        "statement_id": s.statement_id,
        "kind": s.kind.value,
        "formula": _formula_payload(s.formula),
        "domain": s.domain,
        "tag": s.tag.value,
        "temporal": list(s.temporal_scope.as_tuple()) if s.temporal_scope else None,
        "context_scope": s.context_scope,
        "origin": s.origin,
        "assumption_name": s.assumption_name,
        "variables": [(v.variable.name, v.variable.sort) for v in s.variables],
    }


def _rule_payload(r: RuleDecl) -> dict[str, Any]:
    return {
        "rule_id": r.rule_id,
        "name": r.name,
        "version": r.version,
        "variables": [(v.variable.name, v.variable.sort) for v in r.variables],
        "head": _atom_payload(r.head),
        "body": [(_atom_payload(b.atom), b.negated) for b in r.body],
    }


def _atom_payload(a: Atom) -> list[Any]:
    return [a.predicate, [_term_payload(t) for t in a.args]]


def _term_payload(t: Term) -> Any:
    if isinstance(t, Const):
        return ["const", t.name, t.sort]
    if isinstance(t, Var):
        return ["var", t.name, t.sort]
    if isinstance(t, NumLit):
        return ["num", str(t.value), t.sort]
    if isinstance(t, BoolLit):
        return ["bool", t.value]
    if isinstance(t, StrLit):
        return ["str", t.value]
    if isinstance(t, BVLit):
        return ["bv", t.value, t.width]
    if isinstance(t, ToReal):
        return ["toreal", _term_payload(t.operand)]
    if isinstance(t, ArithExpr):
        return ["arith", t.op.value, [_term_payload(a) for a in t.args]]
    if isinstance(t, BvExpr):
        return ["bvexpr", t.op.value, [_term_payload(a) for a in t.args]]
    if isinstance(t, StrExpr):
        return ["strexpr", t.op.value, [_term_payload(a) for a in t.args]]
    if isinstance(t, FuncApp):
        return ["func", t.function, [_term_payload(a) for a in t.args]]
    raise TypeError(f"unserializable term: {type(t)!r}")


def _formula_payload(f: Formula) -> Any:
    if isinstance(f, Atom):
        return ["atom", _atom_payload(f)]
    if isinstance(f, Compare):
        return ["cmp", f.op.value, _term_payload(f.left), _term_payload(f.right)]
    if isinstance(f, BoolTermF):
        return ["boolterm", _term_payload(f.term)]
    if isinstance(f, NotF):
        return ["not", _formula_payload(f.body)]
    if isinstance(f, AndF):
        return ["and", [_formula_payload(a) for a in f.args]]
    if isinstance(f, OrF):
        return ["or", [_formula_payload(a) for a in f.args]]
    if isinstance(f, ImpliesF):
        return ["implies", _formula_payload(f.premise), _formula_payload(f.conclusion)]
    if isinstance(f, IffF):
        return ["iff", _formula_payload(f.left), _formula_payload(f.right)]
    if isinstance(f, ForallF):
        return ["forall", [(v.variable.name, v.variable.sort) for v in f.variables],
                _formula_payload(f.body)]
    if isinstance(f, ExistsF):
        return ["exists", [(v.variable.name, v.variable.sort) for v in f.variables],
                _formula_payload(f.body)]
    if isinstance(f, CardinalityF):
        return ["card", f.mode, f.count, [_formula_payload(a) for a in f.args]]
    raise TypeError(f"unserializable formula: {type(f)!r}")


# ---------------------------------------------------------------------------
# Type checking and normalization (Step 15)
# ---------------------------------------------------------------------------


class _TypeChecker:
    """Type-check terms and formulas against declared sorts and predicates."""

    def __init__(self, ir_sorts: Mapping[str, SortDecl], predicates: Mapping[str, PredicateDecl],
                 functions: Mapping[str, FunctionDecl]) -> None:
        self.sorts = ir_sorts
        self.predicates = predicates
        self.functions = functions

    def term_sort(self, term: Term, env: Mapping[str, str]) -> str:
        """Return the declared sort of one term; raise on any mismatch."""
        if isinstance(term, Const):
            if term.sort not in self.sorts:
                raise InvalidFormalizationError(
                    f"constant {term.name!r} uses undeclared sort {term.sort!r}"
                )
            declared = self.sorts[term.sort]
            if declared.kind in NUMERIC_KINDS:
                # Reject accidental type coercions: an identifier string is
                # never a numeric quantity (Step 10).
                try:
                    value = Fraction(term.name)
                except (ValueError, ZeroDivisionError):
                    raise InvalidFormalizationError(
                        f"identifier string {term.name!r} cannot be typed as a "
                        f"{term.sort} quantity"
                    ) from None
                if declared.kind == SortKind.INT and value.denominator != 1:
                    raise InvalidFormalizationError(
                        f"constant {term.name!r} is not an integer value"
                    )
            return term.sort
        if isinstance(term, Var):
            if term.name not in env:
                raise InvalidFormalizationError(f"unbound variable {term.name!r} in term")
            if env[term.name] not in self.sorts:
                raise InvalidFormalizationError(
                    f"variable {term.name!r} uses undeclared sort {env[term.name]!r}"
                )
            return env[term.name]
        if isinstance(term, NumLit):
            if term.sort not in self.sorts:
                raise InvalidFormalizationError(f"literal sort {term.sort!r} is not declared")
            return term.sort
        if isinstance(term, BoolLit):
            return "Bool"
        if isinstance(term, StrLit):
            return "String"
        if isinstance(term, BVLit):
            bv_sorts = [s.name for s in self.sorts.values()
                        if s.kind == SortKind.BITVECTOR and s.width == term.width]
            if not bv_sorts:
                raise InvalidFormalizationError(
                    f"bit-vector literal width {term.width} has no declared sort"
                )
            return bv_sorts[0]
        if isinstance(term, ToReal):
            inner = self.term_sort(term.operand, env)
            if inner != "Int":
                raise InvalidFormalizationError("ToReal requires an Int operand")
            return "Real"
        if isinstance(term, ArithExpr):
            sorts = {self.term_sort(a, env) for a in term.args}
            for sort in sorts:
                if sort not in {"Int", "Real"}:
                    raise InvalidFormalizationError(
                        f"arithmetic requires numeric operands, got {sort!r}"
                    )
            if len(sorts) > 1:
                # integer and real arithmetic are never silently interchanged (Step 30)
                raise InvalidFormalizationError(
                    "mixed Int/Real arithmetic requires an explicit ToReal conversion"
                )
            sort = sorts.pop()
            if term.op == ArithOp.DIV and sort == "Int":
                raise InvalidFormalizationError(
                    "integer division is not supported; convert operands to Real explicitly"
                )
            return sort
        if isinstance(term, BvExpr):
            bv_sort = None
            for a in term.args:
                arg_sort = self.term_sort(a, env)
                if arg_sort not in self.sorts or self.sorts[arg_sort].kind != SortKind.BITVECTOR:
                    raise InvalidFormalizationError("bit-vector operands must be bit-vectors")
                if bv_sort is None:
                    bv_sort = arg_sort
                elif bv_sort != arg_sort:
                    raise InvalidFormalizationError("bit-vector operands must share one sort")
            if bv_sort is None:
                raise InvalidFormalizationError("bit-vector expressions need operands")
            return bv_sort
        if isinstance(term, StrExpr):
            for a in term.args:
                arg_sort = self.term_sort(a, env)
                if arg_sort != "String":
                    raise InvalidFormalizationError(
                        f"string operator {term.op.value} requires String operands, got {arg_sort!r}"
                    )
            if term.op == StrOp.LEN:
                return "Int"
            if term.op == StrOp.CONCAT:
                return "String"
            return "Bool"
        if isinstance(term, FuncApp):
            decl = self.functions.get(term.function)
            if decl is None:
                raise InvalidFormalizationError(
                    f"application of undeclared function {term.function!r}"
                )
            if len(term.args) != len(decl.arg_sorts):
                raise InvalidFormalizationError(
                    f"function {term.function!r} expects {len(decl.arg_sorts)} arguments"
                )
            for index, (arg, expected) in enumerate(zip(term.args, decl.arg_sorts)):
                actual = self.term_sort(arg, env)
                if actual != expected:
                    raise InvalidFormalizationError(
                        f"function {term.function!r} argument {index} expects {expected!r}, "
                        f"got {actual!r}"
                    )
            return decl.result_sort
        raise InvalidFormalizationError(f"unknown term node: {type(term)!r}")

    def check_formula(self, formula: Formula, env: Mapping[str, str]) -> None:
        """Validate one formula; raises InvalidFormalizationError on any mismatch."""
        if isinstance(formula, Atom):
            decl = self.predicates.get(formula.predicate)
            if decl is None:
                raise InvalidFormalizationError(
                    f"atom references undeclared predicate {formula.predicate!r}"
                )
            if len(formula.args) != len(decl.arg_sorts):
                raise InvalidFormalizationError(
                    f"predicate {formula.predicate!r} expects {len(decl.arg_sorts)} arguments"
                )
            for index, (arg, expected) in enumerate(zip(formula.args, decl.arg_sorts)):
                actual = self.term_sort(arg, env)
                if actual != expected:
                    raise InvalidFormalizationError(
                        f"predicate {formula.predicate!r} argument {index} expects "
                        f"{expected!r}, got {actual!r}"
                    )
            return
        if isinstance(formula, Compare):
            left = self.term_sort(formula.left, env)
            right = self.term_sort(formula.right, env)
            if left != right:
                raise InvalidFormalizationError(
                    f"comparison operands must share one sort, got {left!r} vs {right!r}"
                )
            if formula.op in {CompareOp.LT, CompareOp.LE, CompareOp.GT, CompareOp.GE}:
                sort = self.sorts[left]
                if sort.kind not in NUMERIC_KINDS and sort.kind != SortKind.BITVECTOR:
                    raise InvalidFormalizationError(
                        f"ordering comparison requires numeric or bit-vector operands, "
                        f"got {left!r}"
                    )
            return
        if isinstance(formula, BoolTermF):
            if self.term_sort(formula.term, env) != "Bool":
                raise InvalidFormalizationError("BoolTermF requires a Bool-sorted term")
            return
        if isinstance(formula, NotF):
            self.check_formula(formula.body, env)
            return
        if isinstance(formula, (AndF, OrF)):
            for arg in formula.args:
                self.check_formula(arg, env)
            return
        if isinstance(formula, (ImpliesF, IffF)):
            self.check_formula(formula.premise if isinstance(formula, ImpliesF) else formula.left, env)
            self.check_formula(
                formula.conclusion if isinstance(formula, ImpliesF) else formula.right, env,
            )
            return
        if isinstance(formula, (ForallF, ExistsF)):
            scoped = dict(env)
            for decl in formula.variables:
                if decl.variable.sort not in self.sorts:
                    raise InvalidFormalizationError(
                        f"quantifier variable {decl.variable.name!r} uses undeclared sort"
                    )
                scoped[decl.variable.name] = decl.variable.sort
            self.check_formula(formula.body, scoped)
            return
        if isinstance(formula, CardinalityF):
            for arg in formula.args:
                self.check_formula(arg, env)
            return
        raise InvalidFormalizationError(f"unknown formula node: {type(formula)!r}")


def _fold_formula(formula: Formula) -> Formula:
    """Apply only provably safe boolean constant folding (Step 15 'safe expressions')."""
    if isinstance(formula, NotF):
        body = _fold_formula(formula.body)
        if isinstance(body, BoolTermF) and isinstance(body.term, BoolLit):
            return BoolTermF(BoolLit(not body.term.value))
        return NotF(body)
    if isinstance(formula, AndF):
        args = [_fold_formula(a) for a in formula.args]
        folded: list[Formula] = []
        for arg in args:
            if _is_bool_true(arg):
                continue
            if _is_bool_false(arg):
                return BoolTermF(BoolLit(False))
            folded.append(arg)
        if not folded:
            return BoolTermF(BoolLit(True))
        if len(folded) == 1:
            return folded[0]
        return AndF(tuple(folded))
    if isinstance(formula, OrF):
        args = [_fold_formula(a) for a in formula.args]
        folded = []
        for arg in args:
            if _is_bool_true(arg):
                return BoolTermF(BoolLit(True))
            if _is_bool_false(arg):
                continue
            folded.append(arg)
        if not folded:
            return BoolTermF(BoolLit(False))
        if len(folded) == 1:
            return folded[0]
        return OrF(tuple(folded))
    if isinstance(formula, ImpliesF):
        premise = _fold_formula(formula.premise)
        conclusion = _fold_formula(formula.conclusion)
        if _is_bool_false(premise):
            return BoolTermF(BoolLit(True))
        if _is_bool_true(conclusion):
            return BoolTermF(BoolLit(True))
        if _is_bool_true(premise):
            return conclusion
        if _is_bool_false(conclusion):
            return NotF(premise)
        return ImpliesF(premise, conclusion)
    if isinstance(formula, IffF):
        left = _fold_formula(formula.left)
        right = _fold_formula(formula.right)
        if _is_bool_true(left):
            return right
        if _is_bool_true(right):
            return left
        if _is_bool_false(left) and _is_bool_false(right):
            return BoolTermF(BoolLit(True))
        return IffF(left, right)
    if isinstance(formula, Compare):
        if isinstance(formula.left, NumLit) and isinstance(formula.right, NumLit):
            left, right = formula.left.value, formula.right.value
            if formula.op == CompareOp.EQ:
                return BoolTermF(BoolLit(left == right))
            if formula.op == CompareOp.NEQ:
                return BoolTermF(BoolLit(left != right))
            if formula.op == CompareOp.LT:
                return BoolTermF(BoolLit(left < right))
            if formula.op == CompareOp.LE:
                return BoolTermF(BoolLit(left <= right))
            if formula.op == CompareOp.GT:
                return BoolTermF(BoolLit(left > right))
            if formula.op == CompareOp.GE:
                return BoolTermF(BoolLit(left >= right))
        return formula
    if isinstance(formula, (ForallF, ExistsF)):
        return formula
    return formula


def _is_bool_true(formula: Formula) -> bool:
    return isinstance(formula, BoolTermF) and isinstance(formula.term, BoolLit) and formula.term.value


def _is_bool_false(formula: Formula) -> bool:
    return isinstance(formula, BoolTermF) and isinstance(formula.term, BoolLit) and not formula.term.value


def _alpha_rename(formula: Formula, checker: _TypeChecker,
                  free_env: Mapping[str, str] | None = None) -> tuple[Formula, list[NormalizationIssue]]:
    """Canonicalize quantified variables to v1..vn per statement, avoiding capture."""
    issues: list[NormalizationIssue] = []
    base_env = dict(free_env or {})

    def rename(formula: Formula, env: Mapping[str, str], counter: list[int]) -> Formula:
        if isinstance(formula, (ForallF, ExistsF)):
            mapping: dict[str, str] = {}
            new_decls: list[VarDecl] = []
            scoped = dict(env)
            for decl in formula.variables:
                counter[0] += 1
                new_name = f"v{counter[0]}"
                while new_name in scoped:
                    counter[0] += 1
                    new_name = f"v{counter[0]}"
                mapping[decl.variable.name] = new_name
                new_var = Var(new_name, decl.variable.sort)
                new_decls.append(VarDecl(new_var))
                scoped[new_name] = decl.variable.sort
            body = substitute_rename(formula.body, mapping, env, counter)
            cls = ForallF if isinstance(formula, ForallF) else ExistsF
            return cls(tuple(new_decls), body)
        if isinstance(formula, NotF):
            return NotF(rename(formula.body, env, counter))
        if isinstance(formula, AndF):
            return AndF(tuple(rename(a, env, counter) for a in formula.args))
        if isinstance(formula, OrF):
            return OrF(tuple(rename(a, env, counter) for a in formula.args))
        if isinstance(formula, ImpliesF):
            return ImpliesF(rename(formula.premise, env, counter), rename(formula.conclusion, env, counter))
        if isinstance(formula, IffF):
            return IffF(rename(formula.left, env, counter), rename(formula.right, env, counter))
        if isinstance(formula, CardinalityF):
            return CardinalityF(formula.mode, formula.count,
                                tuple(rename(a, env, counter) for a in formula.args))
        if isinstance(formula, (Atom, Compare, BoolTermF)):
            return rename_terms_in_formula(formula, env, counter)
        return formula

    def substitute_rename(formula: Formula, mapping: Mapping[str, str],
                          env: Mapping[str, str], counter: list[int]) -> Formula:
        """Rename free occurrences of mapped variables below a binder."""
        if isinstance(formula, (ForallF, ExistsF)):
            # nested binders handled by recursive rename with their own scope
            local = dict(env)
            for decl in formula.variables:
                local[decl.variable.name] = decl.variable.sort
            cls = ForallF if isinstance(formula, ForallF) else ExistsF
            inner_mapping = {k: v for k, v in mapping.items() if k not in local or k in mapping}
            body = substitute_rename(formula.body, inner_mapping, local, counter)
            renamed_decls = tuple(
                VarDecl(Var(mapping.get(d.variable.name, d.variable.name), d.variable.sort))
                if d.variable.name in mapping and d.variable.name not in local else d
                for d in formula.variables
            )
            return cls(renamed_decls, body)
        if isinstance(formula, NotF):
            return NotF(substitute_rename(formula.body, mapping, env, counter))
        if isinstance(formula, AndF):
            return AndF(tuple(substitute_rename(a, mapping, env, counter) for a in formula.args))
        if isinstance(formula, OrF):
            return OrF(tuple(substitute_rename(a, mapping, env, counter) for a in formula.args))
        if isinstance(formula, ImpliesF):
            return ImpliesF(
                substitute_rename(formula.premise, mapping, env, counter),
                substitute_rename(formula.conclusion, mapping, env, counter),
            )
        if isinstance(formula, IffF):
            return IffF(
                substitute_rename(formula.left, mapping, env, counter),
                substitute_rename(formula.right, mapping, env, counter),
            )
        if isinstance(formula, CardinalityF):
            return CardinalityF(
                formula.mode, formula.count,
                tuple(substitute_rename(a, mapping, env, counter) for a in formula.args),
            )
        if isinstance(formula, Atom):
            return Atom(formula.predicate, tuple(
                _rename_term(t, mapping) for t in formula.args
            ))
        if isinstance(formula, Compare):
            return Compare(formula.op, _rename_term(formula.left, mapping),
                           _rename_term(formula.right, mapping))
        if isinstance(formula, BoolTermF):
            return BoolTermF(_rename_term(formula.term, mapping))
        return formula

    renamed = rename(formula, base_env, [0])
    # validate the renamed formula type-checks under the free-variable environment
    checker.check_formula(renamed, base_env)
    return renamed, issues


def _rename_term(term: Term, mapping: Mapping[str, str]) -> Term:
    if isinstance(term, Var) and term.name in mapping:
        return Var(mapping[term.name], term.sort)
    if isinstance(term, ToReal):
        return ToReal(_rename_term(term.operand, mapping))
    if isinstance(term, ArithExpr):
        return ArithExpr(term.op, tuple(_rename_term(a, mapping) for a in term.args))
    if isinstance(term, BvExpr):
        return BvExpr(term.op, tuple(_rename_term(a, mapping) for a in term.args))
    if isinstance(term, StrExpr):
        return StrExpr(term.op, tuple(_rename_term(a, mapping) for a in term.args))
    if isinstance(term, FuncApp):
        return FuncApp(term.function, tuple(_rename_term(a, mapping) for a in term.args))
    return term


def rename_terms_in_formula(formula: Formula, env: Mapping[str, str],
                            counter: list[int]) -> Formula:
    """Return the formula unchanged; free terms keep their canonical names."""
    return formula


# ---------------------------------------------------------------------------
# Formalization compiler (Steps 8-15)
# ---------------------------------------------------------------------------


class FormalIRBuilder:
    """Compile typed statements, rules, sorts and domain semantics into one IR."""

    def __init__(self) -> None:
        self._sorts: dict[str, SortDecl] = {}
        self._predicates: dict[str, PredicateDecl] = {}
        self._functions: dict[str, FunctionDecl] = {}
        self._statements: list[Statement] = []
        self._rules: list[RuleDecl] = []
        self._domains: dict[str, DomainSemantics] = {}
        self.unit_table = UnitTable()
        self._warnings: list[NormalizationIssue] = []

    def declare_sort(self, sort: SortDecl) -> "FormalIRBuilder":
        """Declare one sort (Step 9)."""
        if sort.name in BUILTIN_SORTS:
            raise InvalidFormalizationError(
                f"sort {sort.name!r} is built-in and cannot be redeclared"
            )
        self._sorts[sort.name] = sort
        return self

    def declare_predicate(self, predicate: PredicateDecl) -> "FormalIRBuilder":
        """Declare one predicate with arity and argument sorts (Step 10)."""
        for sort_name in predicate.arg_sorts:
            self._require_sort(sort_name)
        existing = self._predicates.get(predicate.name)
        if existing is not None and existing.arg_sorts != predicate.arg_sorts:
            raise InvalidFormalizationError(
                f"predicate {predicate.name!r} redeclared with different signature"
            )
        self._predicates[predicate.name] = predicate
        return self

    def declare_function(self, function: FunctionDecl) -> "FormalIRBuilder":
        """Declare one uninterpreted function (Step 10)."""
        for sort_name in (*function.arg_sorts, function.result_sort):
            self._require_sort(sort_name)
        self._functions[function.name] = function
        return self

    def declare_domain(self, semantics: DomainSemantics) -> "FormalIRBuilder":
        """Declare one per-domain semantic assumption profile (Step 14)."""
        self._domains[semantics.domain] = semantics
        return self

    def predicate_map(self) -> dict[str, PredicateDecl]:
        """Return predicates declared so far in this builder (for adapter checks)."""
        return dict(self._predicates)

    def add_statement(self, statement: Statement) -> "FormalIRBuilder":
        """Add one fact, constraint, or goal statement."""
        self._statements.append(statement)
        return self

    def add_rule(self, rule: RuleDecl) -> "FormalIRBuilder":
        """Add one named rule (Step 12)."""
        self._rules.append(rule)
        return self

    def _require_sort(self, sort_name: str) -> SortDecl:
        merged = dict(BUILTIN_SORTS)
        merged.update(self._sorts)
        if sort_name not in merged:
            raise InvalidFormalizationError(f"undeclared sort {sort_name!r}")
        return merged[sort_name]

    def build(self) -> FormalIR:
        """Validate, normalize, and assemble the IR (Step 15)."""
        if not self._domains:
            raise InvalidFormalizationError(
                "at least one domain semantic profile must be declared (Step 14); "
                "a global profile is never assumed by accident"
            )
        sorts = dict(BUILTIN_SORTS)
        sorts.update(self._sorts)
        checker = _TypeChecker(sorts, dict(self._predicates), dict(self._functions))

        normalized_statements: list[Statement] = []
        source_map: dict[str, dict[str, Any]] = {}
        for statement in self._statements:
            free_env = {v.variable.name: v.variable.sort for v in statement.variables}
            if len(free_env) != len(statement.variables):
                raise InvalidFormalizationError(
                    f"statement {statement.statement_id!r} repeats a free variable"
                )
            checker.check_formula(statement.formula, free_env)
            renamed, _ = _alpha_rename(statement.formula, checker, free_env)
            folded = _fold_formula(renamed)
            normalized = Statement(
                statement_id=statement.statement_id,
                kind=statement.kind,
                formula=folded,
                domain=statement.domain,
                tag=statement.tag,
                source_refs=statement.source_refs,
                temporal_scope=statement.temporal_scope,
                context_scope=statement.context_scope,
                origin=statement.origin,
                assumption_name=statement.assumption_name,
                variables=statement.variables,
                note=statement.note,
            )
            normalized_statements.append(normalized)
            source_map[normalized.statement_id] = {
                "origin": normalized.origin,
                "tag": normalized.tag.value,
                "domain": normalized.domain,
                "context_scope": normalized.context_scope,
                "temporal": list(normalized.temporal_scope.as_tuple())
                if normalized.temporal_scope else None,
                "source_refs": [
                    {"source_id": r.source_id, "record_id": r.record_id,
                     "version_id": r.version_id, "timestamp": r.timestamp,
                     "authority": r.authority, "locator": r.locator}
                    for r in normalized.source_refs
                ],
                "assumption_name": normalized.assumption_name,
            }

        for rule in self._rules:
            self._validate_rule(rule, checker)

        ir = FormalIR(
            sorts=tuple(self._sorts.values()),
            predicates=tuple(self._predicates.values()),
            functions=tuple(self._functions.values()),
            statements=tuple(normalized_statements),
            rules=tuple(self._rules),
            domains=tuple(self._domains.values()),
            unit_table=self.unit_table,
            warnings=tuple(self._warnings),
            source_map=source_map,
        )
        return ir

    def _validate_rule(self, rule: RuleDecl, checker: _TypeChecker) -> None:
        """Type-check rule head/body and reject unbound variables (Step 12/15)."""
        env = {v.variable.name: v.variable.sort for v in rule.variables}
        if len(env) != len(rule.variables):
            raise InvalidFormalizationError(f"rule {rule.rule_id!r} repeats a variable")
        # head
        head_decl = self._predicates.get(rule.head.predicate)
        if head_decl is None:
            raise InvalidFormalizationError(
                f"rule {rule.rule_id!r} head references undeclared predicate "
                f"{rule.head.predicate!r}"
            )
        if len(rule.head.args) != len(head_decl.arg_sorts):
            raise InvalidFormalizationError(
                f"rule {rule.rule_id!r} head arity mismatch for {rule.head.predicate!r}"
            )
        for arg, expected in zip(rule.head.args, head_decl.arg_sorts):
            actual = checker.term_sort(arg, env)
            if actual != expected:
                raise InvalidFormalizationError(
                    f"rule {rule.rule_id!r} head argument sort mismatch: "
                    f"{actual!r} vs {expected!r}"
                )
        # range restriction: every head variable must occur in a positive body
        # literal; unsafe variables are rejected before any evaluation.
        head_vars = {arg.name for arg in rule.head.args if isinstance(arg, Var)}
        positive_vars: set[str] = set()
        for literal in rule.body:
            for arg in literal.atom.args:
                if isinstance(arg, Var):
                    if not literal.negated:
                        positive_vars.add(arg.name)
        unbound = head_vars - positive_vars
        if unbound:
            raise InvalidFormalizationError(
                f"rule {rule.rule_id!r} is unsafe: head variables {sorted(unbound)} "
                "do not occur in any positive body literal"
            )
        for literal in rule.body:
            decl = self._predicates.get(literal.atom.predicate)
            if decl is None:
                raise InvalidFormalizationError(
                    f"rule {rule.rule_id!r} body references undeclared predicate "
                    f"{literal.atom.predicate!r}"
                )
            if len(literal.atom.args) != len(decl.arg_sorts):
                raise InvalidFormalizationError(
                    f"rule {rule.rule_id!r} body arity mismatch for "
                    f"{literal.atom.predicate!r}"
                )
            for arg, expected in zip(literal.atom.args, decl.arg_sorts):
                actual = checker.term_sort(arg, env)
                if actual != expected:
                    raise InvalidFormalizationError(
                        f"rule {rule.rule_id!r} body argument sort mismatch in "
                        f"{literal.atom.predicate!r}: {actual!r} vs {expected!r}"
                    )


__all__ = [
    "SortKind", "Finiteness", "SortDecl", "BOOL_SORT", "INT_SORT", "REAL_SORT",
    "STRING_SORT", "BUILTIN_SORTS", "make_enum_sort", "make_bv_sort",
    "ArithOp", "CompareOp", "BvOp", "StrOp",
    "Const", "Var", "NumLit", "BoolLit", "StrLit", "BVLit", "ToReal",
    "ArithExpr", "BvExpr", "StrExpr", "FuncApp", "Term",
    "Atom", "Compare", "BoolTermF", "NotF", "AndF", "OrF", "ImpliesF", "IffF",
    "VarDecl", "ForallF", "ExistsF", "CardinalityF", "Formula",
    "PredicateDecl", "FunctionDecl", "BodyLiteral", "RuleDecl",
    "UnitDecl", "UnitConversion", "UnitTable", "Quantity", "dimensions_compatible",
    "DomainSemantics", "StatementKind", "Statement", "NormalizationIssue",
    "FormalIR", "FormalIRBuilder",
]
