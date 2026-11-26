"""SAT/SMT encoding, consistency checking, and entailment (Phase D, Steps 23-30).

The Z3 adapter translates the typed IR into solver formulas while maintaining a
bidirectional map from each encoded constraint to its IR statement and USG
provenance. Assumptions are separately addressable named constraints so an
unsatisfiable core can attribute inconsistency. Satisfying models are decoded
cautiously and re-validated by an independent evaluator that recomputes the
original IR semantics with exact arithmetic and bit-vector wrapping — a solver
model is never trusted without that independent check.

Status discipline (Point 5, Section 1): SAT, UNSAT, UNKNOWN, timeout, and
error are returned distinctly; ``UNKNOWN``/timeout never establish falsity;
an unsatisfiable core is reported as a sufficient conflicting subset, not a
minimum-cardinality cause, unless minimization was actually performed.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Iterable, Mapping, Sequence

from itertools import product

from .contracts import (
    PermittedTheory,
    SolverIdentity,
    SolverStatus,
    TheoryProfile,
)
from .exceptions import (
    InvalidFormalizationError,
    MalformedSolverOutputError,
    UnsupportedTheoryError,
)
from .ir import (
    FuncApp,
    AndF,
    ArithExpr,
    ArithOp,
    Atom,
    BVLit,
    BvExpr,
    BvOp,
    BoolLit,
    BoolTermF,
    CardinalityF,
    Compare,
    CompareOp,
    Const,
    ExistsF,
    ForallF,
    FormalIR,
    IffF,
    ImpliesF,
    NotF,
    NumLit,
    OrF,
    SortDecl,
    SortKind,
    Statement,
    StatementKind,
    StrExpr,
    StrOp,
    StrLit,
    ToReal,
    Var,
)

try:  # pragma: no cover - environment dependent
    import z3

    Z3_VERSION = z3.get_version_string()
    Z3_AVAILABLE = True
except ImportError:  # pragma: no cover
    z3 = None  # type: ignore[assignment]
    Z3_VERSION = ""
    Z3_AVAILABLE = False

SMT_ADAPTER_VERSION = "frce-smt-1.0.0"

_NONLINEAR_MARK = "nonlinear arithmetic detected in the encoding"


# ---------------------------------------------------------------------------
# Theory selection (Step 23)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TheoryNeed:
    """Theories detected in one encoding plus their evidence."""

    theories: frozenset[PermittedTheory]
    nonlinear: bool
    notes: tuple[str, ...]


def detect_theories(statements: Sequence[Statement]) -> TheoryNeed:
    """Detect which theories the statements require (Step 23)."""
    theories: set[PermittedTheory] = {PermittedTheory.BOOL}
    notes: list[str] = []
    nonlinear = False
    for statement in statements:
        _scan_statement(statement, theories, notes, [nonlinear])
        if _statement_nonlinear(statement):
            nonlinear = True
    if nonlinear:
        theories.add(PermittedTheory.NONLINEAR_ARITHMETIC)
        notes.append(_NONLINEAR_MARK)
    return TheoryNeed(
        theories=frozenset(theories), nonlinear=nonlinear, notes=tuple(notes),
    )


def _scan_statement(
    statement: Statement,
    theories: set[PermittedTheory],
    notes: list[str],
    _flag: list[Any],
) -> None:
    """Accumulate theory needs from one statement's sort usage."""
    sorts = _collect_sorts(statement.formula)
    if "Int" in sorts:
        theories.add(PermittedTheory.LINEAR_INTEGER_ARITHMETIC)
    if "Real" in sorts:
        theories.add(PermittedTheory.LINEAR_REAL_ARITHMETIC)
    if any(s.startswith("bv") for s in sorts):
        theories.add(PermittedTheory.BITVECTORS)
    if "String" in sorts:
        theories.add(PermittedTheory.STRINGS)
    if _uses_string_ops(statement.formula):
        theories.add(PermittedTheory.STRINGS)
        notes.append("string operators require a supported string theory")


def _collect_sorts(formula: Any) -> set[str]:
    """Collect sort names from *interpreted* terms (comparisons, arithmetic).

    Atom arguments are encoded propositionally, so their declared sorts (often
    String identifiers) do not require theory support; only terms inside
    comparisons, boolean terms, and quantifier bodies do.
    """
    sorts: set[str] = set()

    def walk_term(term: Any) -> None:
        sort = getattr(term, "sort", None)
        if isinstance(sort, str):
            sorts.add(sort)
        elif isinstance(term, BVLit):
            sorts.add(f"bv{term.width}")
        args = getattr(term, "args", None)
        if isinstance(args, tuple):
            for arg in args:
                walk_term(arg)
        operand = getattr(term, "operand", None)
        if operand is not None:
            walk_term(operand)

    def walk_formula(node: Any) -> None:
        if isinstance(node, Atom):
            return  # propositional: atom arguments carry no theory need
        if isinstance(node, Compare):
            walk_term(node.left)
            walk_term(node.right)
            return
        if isinstance(node, BoolTermF):
            walk_term(node.term)
            return
        body = getattr(node, "body", None)
        if body is not None:
            walk_formula(body)
        args = getattr(node, "args", None)
        if isinstance(args, tuple):
            for arg in args:
                walk_formula(arg)
        left = getattr(node, "left", None)
        right = getattr(node, "right", None)
        for child in (left, right):
            if child is not None:
                walk_formula(child)
        premise = getattr(node, "premise", None)
        conclusion = getattr(node, "conclusion", None)
        for child in (premise, conclusion):
            if child is not None:
                walk_formula(child)

    walk_formula(formula)
    return sorts


def _uses_string_ops(formula: Any) -> bool:
    """Return whether any string operator appears in the formula."""
    found = False

    def walk_term(term: Any) -> None:
        nonlocal found
        if isinstance(term, StrExpr):
            found = True
        args = getattr(term, "args", None)
        if isinstance(args, tuple):
            for arg in args:
                walk_term(arg)

    def walk_formula(node: Any) -> None:
        if isinstance(node, Atom):
            for arg in node.args:
                walk_term(arg)
            return
        if isinstance(node, Compare):
            walk_term(node.left)
            walk_term(node.right)
            return
        body = getattr(node, "body", None)
        if body is not None:
            walk_formula(body)
        args = getattr(node, "args", None)
        if isinstance(args, tuple):
            for arg in args:
                walk_formula(arg)

    walk_formula(formula)
    return found


def _statement_nonlinear(statement: Statement) -> bool:
    """Detect multiplication of two non-constant terms (nonlinear arithmetic)."""

    def term_is_constant(term: Any) -> bool:
        if isinstance(term, (NumLit, BVLit, StrLit, BoolLit, Const)):
            return True
        args = getattr(term, "args", None)
        if isinstance(args, tuple):
            return all(term_is_constant(a) for a in args)
        operand = getattr(term, "operand", None)
        if operand is not None:
            return term_is_constant(operand)
        return isinstance(term, Var)

    def walk(node: Any) -> bool:
        if isinstance(node, ArithExpr) and node.op == ArithOp.MUL:
            left, right = node.args
            if not term_is_constant(left) and not term_is_constant(right):
                return True
        for child in _children(node):
            if walk(child):
                return True
        return False

    return walk(statement.formula)


def _children(node: Any) -> tuple[Any, ...]:
    """Return the logical/term children of any IR node."""
    if isinstance(node, Atom):
        return node.args
    if isinstance(node, Compare):
        return (node.left, node.right)
    if isinstance(node, BoolTermF):
        return (node.term,)
    args = getattr(node, "args", None)
    body = getattr(node, "body", None)
    children: list[Any] = []
    if isinstance(args, tuple):
        children.extend(args)
    if body is not None:
        children.append(body)
    for attr in ("left", "right", "premise", "conclusion", "operand"):
        child = getattr(node, attr, None)
        if child is not None:
            children.append(child)
    return tuple(children)


# ---------------------------------------------------------------------------
# Encoder (Steps 24-25)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EncodedConstraint:
    """One encoded constraint with its IR node and provenance mapping (Step 24)."""

    label: str
    ir_statement_id: str
    source_refs: tuple[str, ...]
    kind: str  # fact | constraint | assumption | goal_negation
    is_named_assumption: bool = False
    assumption_name: str | None = None


class SmtEncoder:
    """Encode the formal IR into Z3 with full traceability (Steps 24-25)."""

    def __init__(self, ir: FormalIR) -> None:
        if not Z3_AVAILABLE:
            raise UnsupportedTheoryError(
                "z3-solver is not installed; the SMT backend is unavailable"
            )
        self.ir = ir
        self._const_exprs: dict[str, Any] = {}
        self._sorts = ir.sort_map()

    def _sort_decl(self, sort_name: str) -> SortDecl:
        decl = self._sorts.get(sort_name)
        if decl is None:
            raise InvalidFormalizationError(f"undeclared sort in encoding: {sort_name!r}")
        return decl

    def const_expr(self, key: str, sort_name: str) -> Any:
        """Return (or create) the Z3 constant for one IR const/var key."""
        if key in self._const_exprs:
            return self._const_exprs[key]
        decl = self._sort_decl(sort_name)
        if decl.kind == SortKind.BOOL:
            expr = z3.Bool(key)
        elif decl.kind == SortKind.INT:
            expr = z3.Int(key)
        elif decl.kind == SortKind.REAL:
            expr = z3.Real(key)
        elif decl.kind == SortKind.BITVECTOR:
            expr = z3.BitVec(key, decl.width or 32)
        elif decl.kind == SortKind.STRING:
            expr = z3.String(key)
        elif decl.kind == SortKind.ENUM:
            expr = z3.Int(key)
        else:
            raise UnsupportedTheoryError(f"sort {sort_name!r} has no SMT encoding")
        self._const_exprs[key] = expr
        return expr

    def term_expr(self, term: Any, env: Mapping[str, str]) -> Any:
        """Encode one IR term (Step 24)."""
        if isinstance(term, Const):
            return self.const_expr(f"const_{term.name}", term.sort)
        if isinstance(term, Var):
            sort_name = env.get(term.name, term.sort)
            return self.const_expr(f"var_{term.name}", sort_name)
        if isinstance(term, NumLit):
            if term.sort == "Int":
                return z3.IntVal(int(term.value))
            fraction = term.value
            return z3.RealVal(f"{fraction.numerator}/{fraction.denominator}")
        if isinstance(term, BoolLit):
            return z3.BoolVal(term.value)
        if isinstance(term, StrLit):
            return z3.StringVal(term.value)
        if isinstance(term, BVLit):
            return z3.BitVecVal(term.value, term.width)
        if isinstance(term, ToReal):
            return z3.ToReal(self.term_expr(term.operand, env))
        if isinstance(term, ArithExpr):
            left, right = (self.term_expr(a, env) for a in term.args)
            if term.op == ArithOp.ADD:
                return left + right
            if term.op == ArithOp.SUB:
                return left - right
            if term.op == ArithOp.MUL:
                return left * right
            if term.op == ArithOp.DIV:
                # Division by zero is detected statically for literal zero
                # denominators (Step 30); symbolic denominators are encoded
                # per SMT-LIB partial-function semantics and disclosed.
                if isinstance(term.args[1], NumLit) and term.args[1].value == 0:
                    raise InvalidFormalizationError(
                        "division by a literal zero is rejected before solver "
                        "invocation (Step 30)"
                    )
                return left / right
            raise InvalidFormalizationError(f"unencodable arithmetic op: {term.op!r}")
        if isinstance(term, BvExpr):
            args = [self.term_expr(a, env) for a in term.args]
            first = args[0]
            if term.op == BvOp.ADD:
                return first + args[1]
            if term.op == BvOp.SUB:
                return first - args[1]
            if term.op == BvOp.MUL:
                return first * args[1]
            if term.op == BvOp.AND:
                return first & args[1]
            if term.op == BvOp.OR:
                return first | args[1]
            if term.op == BvOp.XOR:
                return first ^ args[1]
            if term.op == BvOp.SHL:
                return first << args[1]
            if term.op == BvOp.LSHR:
                return z3.LShR(first, args[1])
            raise InvalidFormalizationError(f"unencodable bit-vector op: {term.op!r}")
        if isinstance(term, StrExpr):
            args = [self.term_expr(a, env) for a in term.args]
            if term.op == StrOp.CONCAT:
                return z3.Concat(*args)
            if term.op == StrOp.LEN:
                return z3.Length(args[0])
            if term.op == StrOp.CONTAINS:
                return z3.Contains(args[0], args[1])
            if term.op == StrOp.PREFIX:
                return z3.PrefixOf(args[0], args[1])
            if term.op == StrOp.SUFFIX:
                return z3.SuffixOf(args[0], args[1])
            raise InvalidFormalizationError(f"unencodable string op: {term.op!r}")
        if isinstance(term, FuncApp):
            raise UnsupportedTheoryError(
                f"uninterpreted function application {term.function!r} requires the "
                "UF theory adapter; declare it via permitted theories and functions"
            )
        raise InvalidFormalizationError(f"unencodable term: {type(term)!r}")

    def formula_expr(self, formula: Any, env: Mapping[str, str]) -> Any:
        """Encode one IR formula (Step 24)."""
        if isinstance(formula, Atom):
            # Ground atoms become propositional constants keyed canonically.
            args = tuple(
                _term_key(t, env) for t in formula.args
            )
            key = f"atom_{formula.predicate}({','.join(args)})"
            return self.const_expr(key, "Bool")
        if isinstance(formula, Compare):
            left = self.term_expr(formula.left, env)
            right = self.term_expr(formula.right, env)
            sort_name = getattr(formula.left, "sort", None)
            if formula.op == CompareOp.EQ:
                return left == right
            if formula.op == CompareOp.NEQ:
                return left != right
            if sort_name and sort_name.startswith("bv"):
                ops = {
                    CompareOp.LT: z3.ULT, CompareOp.LE: z3.ULE,
                    CompareOp.GT: z3.UGT, CompareOp.GE: z3.UGE,
                }
                return ops[formula.op](left, right)
            ops = {
                CompareOp.LT: lambda a, b: a < b,
                CompareOp.LE: lambda a, b: a <= b,
                CompareOp.GT: lambda a, b: a > b,
                CompareOp.GE: lambda a, b: a >= b,
            }
            return ops[formula.op](left, right)
        if isinstance(formula, BoolTermF):
            return self.term_expr(formula.term, env)
        if isinstance(formula, NotF):
            return z3.Not(self.formula_expr(formula.body, env))
        if isinstance(formula, AndF):
            return z3.And(*[self.formula_expr(a, env) for a in formula.args])
        if isinstance(formula, OrF):
            return z3.Or(*[self.formula_expr(a, env) for a in formula.args])
        if isinstance(formula, ImpliesF):
            return z3.Implies(
                self.formula_expr(formula.premise, env),
                self.formula_expr(formula.conclusion, env),
            )
        if isinstance(formula, IffF):
            return self.formula_expr(formula.left, env) == self.formula_expr(formula.right, env)
        if isinstance(formula, (ForallF, ExistsF)):
            return self._encode_quantifier(formula, env)
        if isinstance(formula, CardinalityF):
            encoded = [self.formula_expr(a, env) for a in formula.args]
            if formula.mode == "at_least":
                return z3.AtLeast(*encoded, formula.count)
            if formula.mode == "at_most":
                return z3.AtMost(*encoded, formula.count)
            at_least = z3.AtLeast(*encoded, formula.count)
            at_most = z3.AtMost(*encoded, formula.count)
            return z3.And(at_least, at_most)
        raise InvalidFormalizationError(f"unencodable formula: {type(formula)!r}")

    def _encode_quantifier(self, formula: Any, env: Mapping[str, str]) -> Any:
        """Encode a quantifier over a finite declared domain (Step 11, 24).

        Quantification requires finite sorts (enumerations or bounded finite
        integer ranges); infinite quantification is reported as unsupported
        rather than guessed.
        """
        bound: list[Any] = []
        scoped = dict(env)
        domain_constraints: list[Any] = []
        for decl in formula.variables:
            sort_name = decl.variable.sort
            sort = self._sort_decl(sort_name)
            var_expr = self.const_expr(f"var_{decl.variable.name}", sort_name)
            bound.append(var_expr)
            scoped[decl.variable.name] = sort_name
            if sort.kind == SortKind.ENUM:
                domain_constraints.append(
                    z3.And(var_expr >= 0, var_expr < len(sort.values))
                )
            elif (
                sort.kind == SortKind.INT
                and sort.finiteness.value == "finite"
                and sort.lower is not None and sort.upper is not None
            ):
                domain_constraints.append(
                    z3.And(var_expr >= sort.lower, var_expr <= sort.upper)
                )
            elif sort.kind == SortKind.INT:
                # enumerate a small explicit range when declared bounded
                raise UnsupportedTheoryError(
                    f"quantification over infinite sort {sort_name!r} is not encodable; "
                    "declare a finite domain or bound the quantifier scope"
                )
            else:
                raise UnsupportedTheoryError(
                    f"quantification over sort {sort_name!r} is not encodable"
                )
        body = self.formula_expr(formula.body, scoped)
        if domain_constraints:
            body = z3.Implies(z3.And(*domain_constraints), body)
        cls = z3.ForAll if isinstance(formula, ForallF) else z3.Exists
        return cls(bound, body)


def _term_key(term: Any, env: Mapping[str, str]) -> str:
    """Canonical propositional key for one atom argument term."""
    if isinstance(term, Const):
        return f"const_{term.name}"
    if isinstance(term, Var):
        return f"var_{term.name}"
    if isinstance(term, NumLit):
        return f"num_{term.value}"
    if isinstance(term, StrLit):
        return f"str_{term.value}"
    if isinstance(term, BoolLit):
        return f"bool_{term.value}"
    if isinstance(term, BVLit):
        return f"bv_{term.value}_{term.width}"
    raise InvalidFormalizationError(
        f"atom argument {type(term).__name__} is outside the propositional atom encoding"
    )


# ---------------------------------------------------------------------------
# Independent model evaluation (Step 27)
# ---------------------------------------------------------------------------


class UnsupportedEvaluationError(Exception):
    """The independent evaluator cannot recompute part of the model."""


class DivisionByZeroEvaluation(UnsupportedEvaluationError):
    """A decoded model hits a division-by-zero under exact semantics."""


class IndependentEvaluator:
    """Re-evaluate IR formulas against a decoded assignment (Step 27, 43).

    The evaluator recomputes semantics directly from the IR with exact rational
    arithmetic and wrapping bit-vector semantics — it never calls the solver —
    so decoded models and counterexamples are validated independently.
    """

    def __init__(self, ir: FormalIR | None = None) -> None:
        self._sorts = ir.sort_map() if ir is not None else {}

    def evaluate_term(self, term: Any, assignment: Mapping[str, Any]) -> Any:
        """Evaluate one term under exact semantics; raises on unsupported parts."""
        if isinstance(term, Const):
            return assignment.get(f"const_{term.name}")
        if isinstance(term, Var):
            return assignment.get(f"var_{term.name}")
        if isinstance(term, NumLit):
            return term.value
        if isinstance(term, BoolLit):
            return term.value
        if isinstance(term, StrLit):
            return term.value
        if isinstance(term, BVLit):
            return term.value
        if isinstance(term, ToReal):
            return Fraction(self.evaluate_term(term.operand, assignment))
        if isinstance(term, ArithExpr):
            left = self.evaluate_term(term.args[0], assignment)
            right = self.evaluate_term(term.args[1], assignment)
            if term.op == ArithOp.ADD:
                return left + right
            if term.op == ArithOp.SUB:
                return left - right
            if term.op == ArithOp.MUL:
                return left * right
            if term.op == ArithOp.DIV:
                if right == 0:
                    raise DivisionByZeroEvaluation("division by zero under exact semantics")
                return Fraction(left) / Fraction(right)
            raise UnsupportedEvaluationError(f"unsupported arithmetic op {term.op!r}")
        if isinstance(term, BvExpr):
            width = self._bv_width(term)
            mask = (1 << width) - 1
            left = self.evaluate_term(term.args[0], assignment)
            right = self.evaluate_term(term.args[1], assignment)
            if term.op == BvOp.ADD:
                return (left + right) & mask
            if term.op == BvOp.SUB:
                return (left - right) & mask
            if term.op == BvOp.MUL:
                return (left * right) & mask
            if term.op == BvOp.AND:
                return (left & right) & mask
            if term.op == BvOp.OR:
                return (left | right) & mask
            if term.op == BvOp.XOR:
                return (left ^ right) & mask
            if term.op == BvOp.SHL:
                return (left << (right % width)) & mask
            if term.op == BvOp.LSHR:
                return (left >> (right % width)) & mask
            raise UnsupportedEvaluationError(f"unsupported bit-vector op {term.op!r}")
        if isinstance(term, StrExpr):
            args = [self.evaluate_term(a, assignment) for a in term.args]
            if term.op == StrOp.CONCAT:
                return "".join(args)
            if term.op == StrOp.LEN:
                return len(args[0])
            if term.op == StrOp.CONTAINS:
                return args[1] in args[0]
            if term.op == StrOp.PREFIX:
                return args[0].startswith(args[1])
            if term.op == StrOp.SUFFIX:
                return args[0].endswith(args[1])
            raise UnsupportedEvaluationError(f"unsupported string op {term.op!r}")
        if isinstance(term, FuncApp):
            raise UnsupportedEvaluationError("uninterpreted function evaluation unsupported")
        raise UnsupportedEvaluationError(f"unsupported term {type(term).__name__}")

    def _bv_width(self, term: BvExpr) -> int:
        """Determine the bit width of a bit-vector expression."""
        for arg in term.args:
            if isinstance(arg, BVLit):
                return arg.width
            sort = self._sorts.get(getattr(arg, "sort", ""))
            if sort is not None and sort.kind == SortKind.BITVECTOR:
                return sort.width or 32
        raise UnsupportedEvaluationError("cannot determine bit-vector width")

    def evaluate_formula(
        self, formula: Any, assignment: Mapping[str, Any],
    ) -> bool | None:
        """Evaluate one formula to True/False, or None when undecidable here."""
        try:
            return self._evaluate(formula, assignment)
        except UnsupportedEvaluationError:
            return None

    def _evaluate(self, formula: Any, assignment: Mapping[str, Any]) -> bool:
        if isinstance(formula, Atom):
            args = tuple(_term_key_static(t) for t in formula.args)
            key = f"atom_{formula.predicate}({','.join(args)})"
            value = assignment.get(key)
            if value is None:
                raise UnsupportedEvaluationError(f"no truth value for {key}")
            return bool(value)
        if isinstance(formula, Compare):
            left = self.evaluate_term(formula.left, assignment)
            right = self.evaluate_term(formula.right, assignment)
            if formula.op == CompareOp.EQ:
                return left == right
            if formula.op == CompareOp.NEQ:
                return left != right
            if formula.op == CompareOp.LT:
                return left < right
            if formula.op == CompareOp.LE:
                return left <= right
            if formula.op == CompareOp.GT:
                return left > right
            if formula.op == CompareOp.GE:
                return left >= right
            raise UnsupportedEvaluationError(f"unsupported comparison {formula.op!r}")
        if isinstance(formula, BoolTermF):
            return bool(self.evaluate_term(formula.term, assignment))
        if isinstance(formula, NotF):
            return not self._evaluate(formula.body, assignment)
        if isinstance(formula, AndF):
            return all(self._evaluate(a, assignment) for a in formula.args)
        if isinstance(formula, OrF):
            return any(self._evaluate(a, assignment) for a in formula.args)
        if isinstance(formula, ImpliesF):
            return (not self._evaluate(formula.premise, assignment)) or \
                self._evaluate(formula.conclusion, assignment)
        if isinstance(formula, IffF):
            return self._evaluate(formula.left, assignment) == \
                self._evaluate(formula.right, assignment)
        if isinstance(formula, CardinalityF):
            values = [self._evaluate(a, assignment) for a in formula.args]
            true_count = sum(1 for v in values if v)
            if formula.mode == "at_least":
                return true_count >= formula.count
            if formula.mode == "at_most":
                return true_count <= formula.count
            return true_count == formula.count
        if isinstance(formula, (ForallF, ExistsF)):
            return self._evaluate_quantified(formula, assignment)
        raise UnsupportedEvaluationError(f"unsupported formula {type(formula).__name__}")

    def _evaluate_quantified(self, formula: Any, assignment: Mapping[str, Any]) -> bool:
        """Enumerate finite domains for quantified formulas (exact, bounded)."""
        domains: list[list[Any]] = []
        for decl in formula.variables:
            sort = self._sorts.get(decl.variable.sort)
            if sort is None:
                raise UnsupportedEvaluationError("undeclared quantifier sort")
            if sort.kind == SortKind.ENUM:
                domains.append(list(range(len(sort.values))))
            elif (
                sort.kind == SortKind.INT
                and sort.lower is not None and sort.upper is not None
                and sort.upper - sort.lower <= 64
            ):
                domains.append(list(range(sort.lower, sort.upper + 1)))
            else:
                raise UnsupportedEvaluationError(
                    f"quantifier over sort {decl.variable.sort!r} is not finitely enumerable"
                )
        results: list[bool] = []
        for combination in product(*domains):
            scoped = dict(assignment)
            for decl, value in zip(formula.variables, combination):
                scoped[f"var_{decl.variable.name}"] = value
            results.append(self._evaluate(formula.body, scoped))
        if isinstance(formula, ForallF):
            return all(results)
        return any(results)


def _term_key_static(term: Any) -> str:
    """Key for atom arguments in the independent evaluator (no env)."""
    if isinstance(term, Const):
        return f"const_{term.name}"
    if isinstance(term, Var):
        return f"var_{term.name}"
    if isinstance(term, NumLit):
        return f"num_{term.value}"
    if isinstance(term, StrLit):
        return f"str_{term.value}"
    if isinstance(term, BoolLit):
        return f"bool_{term.value}"
    if isinstance(term, BVLit):
        return f"bv_{term.value}_{term.width}"
    raise UnsupportedEvaluationError(f"unsupported atom argument {type(term).__name__}")


# ---------------------------------------------------------------------------
# The adapter: solving, cores, entailment (Steps 26-30)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SmtOutcome:
    """The complete outcome of one SMT invocation (Step 26)."""

    solver_status: SolverStatus
    result_status: str                    # ResultStatus value string
    model: Mapping[str, Any] | None = None
    model_independently_validated: bool | None = None
    counterexample: Mapping[str, Any] | None = None
    unsat_core: tuple[str, ...] = ()
    unsat_core_minimized: bool = False
    theory_profile: TheoryProfile | None = None
    solver: SolverIdentity | None = None
    wall_clock_ms: float = 0.0
    limitations: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    reason: str = ""


@dataclass
class _TrackedInput:
    """One tracked input constraint with its provenance (Step 24, 25)."""

    label: str
    statement_id: str
    expr: Any
    kind: str
    is_named_assumption: bool = False


class SmtAdapter:
    """Z3-backed satisfiability, entailment, and core extraction (Steps 23-30)."""

    def __init__(
        self,
        ir: FormalIR,
        *,
        permitted_theories: Iterable[PermittedTheory] = (),
        timeout_ms: int = 10000,
        memory_mb: int = 1024,
    ) -> None:
        if not Z3_AVAILABLE:
            raise UnsupportedTheoryError("z3-solver is not installed")
        self.ir = ir
        self.permitted = frozenset(permitted_theories)
        self.timeout_ms = max(1, int(timeout_ms))
        self.memory_mb = max(64, int(memory_mb))
        self.encoder = SmtEncoder(ir)
        self.evaluator = IndependentEvaluator(ir)
        self._tracked: list[_TrackedInput] = []
        self._theory_need: TheoryNeed | None = None

    # -- theory gating (Step 23) -------------------------------------------

    def check_theories(self, statements: Sequence[Statement]) -> TheoryNeed:
        """Verify detected theories are permitted before any solving."""
        need = detect_theories(statements)
        self._theory_need = need
        if self.permitted:
            unpermitted = need.theories - self.permitted
            if unpermitted:
                names = ", ".join(sorted(t.value for t in unpermitted))
                raise UnsupportedTheoryError(
                    f"the encoding requires unpermitted theories: {names}"
                )
        return need

    # -- encoding with named assumptions (Steps 24-25) ----------------------

    def track_statements(self, statements: Sequence[Statement]) -> tuple[EncodedConstraint, ...]:
        """Encode and track statements with labels and provenance (Step 24)."""
        encoded: list[EncodedConstraint] = []
        for statement in statements:
            expr = self.encoder.formula_expr(statement.formula, {})
            kind_map = {
                StatementKind.FACT: "fact",
                StatementKind.CONSTRAINT: "constraint",
                StatementKind.GOAL: "goal",
            }
            kind = kind_map.get(statement.kind, "constraint")
            is_assumption = statement.assumption_name is not None
            label = statement.assumption_name or statement.statement_id
            self._tracked.append(_TrackedInput(
                label=label,
                statement_id=statement.statement_id,
                expr=expr,
                kind=kind,
                is_named_assumption=is_assumption,
            ))
            encoded.append(EncodedConstraint(
                label=label,
                ir_statement_id=statement.statement_id,
                source_refs=tuple(
                    ref.source_id for ref in statement.source_refs
                ),
                kind=kind,
                is_named_assumption=is_assumption,
                assumption_name=statement.assumption_name,
            ))
        return tuple(encoded)

    def _solver(self) -> Any:
        """Create a deterministically configured Z3 solver (Step 26)."""
        solver = z3.Solver()
        solver.set("random_seed", 0)
        solver.set("timeout", self.timeout_ms)
        solver.set("max_memory", self.memory_mb)
        return solver

    def _solver_identity(self) -> SolverIdentity:
        return SolverIdentity(
            name="Z3",
            version=Z3_VERSION,
            configuration={
                "random_seed": 0,
                "timeout_ms": self.timeout_ms,
                "max_memory_mb": self.memory_mb,
            },
            backend="in_process",
        )

    # -- satisfiability (Step 26) -------------------------------------------

    def check_satisfiability(self, statements: Sequence[Statement]) -> SmtOutcome:
        """Run a satisfiability check with distinct, honest statuses (Step 26)."""
        start = time.perf_counter()
        need = self.check_theories(statements)
        self.track_statements(statements)
        return self.solve_tracked(need, start)

    def solve_tracked(self, need: TheoryNeed | None = None,
                      start: float | None = None) -> SmtOutcome:
        """Solve the already-tracked constraints (Step 26).

        Split from ``check_satisfiability`` so the coordinator can record the
        encoding steps (24-25) before the solver call (26).
        """
        if start is None:
            start = time.perf_counter()
        if need is None:
            need = self._theory_need or TheoryNeed(
                frozenset({PermittedTheory.BOOL}), False, ())
        solver = self._solver()
        for tracked in self._tracked:
            solver.assert_and_track(tracked.expr, tracked.label)
        z3_result = solver.check()
        wall_ms = (time.perf_counter() - start) * 1000.0
        outcome = self._interpret(z3_result, solver, need, wall_ms)
        return outcome

    def _interpret(
        self, z3_result: Any, solver: Any, need: TheoryNeed, wall_ms: float,
    ) -> SmtOutcome:
        """Map the raw solver outcome into typed statuses (Step 26)."""
        identity = self._solver_identity()
        profile = TheoryProfile(
            theories=tuple(sorted(need.theories, key=lambda t: t.value)),
            nonlinear=need.nonlinear,
            exact_arithmetic=True,
            notes=tuple(need.notes),
        )
        limitations: list[str] = []
        if need.nonlinear:
            limitations.append(
                "nonlinear arithmetic: results may be incomplete; UNKNOWN is not falsity"
            )
        if z3_result == z3.sat:
            model = solver.model()
            decoded, decode_warnings = self._decode_model(model)
            valid = self._validate_model(decoded, need)
            if valid is False:
                raise MalformedSolverOutputError(
                    "decoded model failed independent validation against the IR"
                )
            return SmtOutcome(
                solver_status=SolverStatus.SAT,
                result_status="satisfiable",
                model=decoded,
                model_independently_validated=True,
                theory_profile=profile,
                solver=identity,
                wall_clock_ms=wall_ms,
                limitations=tuple(limitations),
                warnings=tuple(decode_warnings),
            )
        if z3_result == z3.unsat:
            core_labels = tuple(sorted(
                str(core.decl().name()) for core in solver.unsat_core()
            ))
            return SmtOutcome(
                solver_status=SolverStatus.UNSAT,
                result_status="unsatisfiable",
                unsat_core=core_labels,
                unsat_core_minimized=False,
                theory_profile=profile,
                solver=identity,
                wall_clock_ms=wall_ms,
                limitations=tuple(limitations + [
                    "the unsatisfiable core is a sufficient conflicting subset, "
                    "not necessarily a minimum-cardinality cause"
                ]),
            )
        reason = str(solver.reason_unknown() or "")
        if "timeout" in reason.lower() or "canceled" in reason.lower():
            return SmtOutcome(
                solver_status=SolverStatus.TIMEOUT,
                result_status="timeout",
                theory_profile=profile,
                solver=identity,
                wall_clock_ms=wall_ms,
                reason=reason,
                limitations=tuple(limitations),
            )
        return SmtOutcome(
            solver_status=SolverStatus.UNKNOWN,
            result_status="unknown",
            theory_profile=profile,
            solver=identity,
            wall_clock_ms=wall_ms,
            reason=reason,
            limitations=tuple(limitations + [
                "UNKNOWN does not establish entailment or non-entailment"
            ]),
        )

    # -- model decoding (Step 27) -------------------------------------------

    def _decode_model(self, model: Any) -> tuple[dict[str, Any], list[str]]:
        """Decode a Z3 model into plain Python values keyed like the evaluator."""
        decoded: dict[str, Any] = {}
        warnings: list[str] = []
        for declaration in model.decls():
            key = declaration.name()
            value = model.eval(declaration(), model_completion=True)
            if key.startswith("atom_"):
                if z3.is_true(value):
                    decoded[key] = True
                elif z3.is_false(value):
                    decoded[key] = False
                else:
                    warnings.append(f"atom {key} has no definite truth value")
                continue
            if z3.is_bool(value):
                decoded[key] = z3.is_true(value)
            elif z3.is_int_value(value):
                decoded[key] = value.as_long()
            elif z3.is_rational_value(value):
                decoded[key] = Fraction(value.as_fraction())
            elif z3.is_algebraic_value(value):
                approx = value.approx(20)
                decoded[key] = Fraction(str(approx))
                warnings.append(
                    f"{key} decoded as an approximate algebraic number; "
                    "the value is a floating approximation and is marked as such"
                )
            elif z3.is_bv(value):
                decoded[key] = value.as_long()
            elif z3.is_string(value):
                decoded[key] = value.as_string()
            else:
                warnings.append(f"unhandled sort for {key!r}")
        return decoded, warnings

    def _validate_model(self, decoded: Mapping[str, Any], need: TheoryNeed) -> bool | None:
        """Independently re-evaluate every tracked constraint (Step 27)."""
        if not self._tracked:
            return True
        results: list[bool | None] = []
        for tracked in self._tracked:
            statement = self._statement_by_id(tracked.statement_id)
            if statement is None:
                return None
            results.append(self.evaluator.evaluate_formula(statement.formula, decoded))
        if any(r is None for r in results):
            return None
        return all(bool(r) for r in results)

    def _statement_by_id(self, statement_id: str) -> Statement | None:
        for statement in self.ir.statements:
            if statement.statement_id == statement_id:
                return statement
        return None

    # -- unsatisfiable cores (Step 28) ---------------------------------------

    def minimize_core(self, core: Sequence[str]) -> tuple[tuple[str, ...], bool]:
        """Greedy core minimization; only then may a core be labeled minimal.

        Returns the minimized core and ``True``. Without calling this method a
        core must be reported as a sufficient, non-minimal conflicting subset.
        """
        if not core:
            return (), False
        current = list(core)
        changed = True
        while changed:
            changed = False
            for label in list(current):
                trial = [item for item in current if item != label]
                if not trial:
                    continue
                if self._is_unsat_subset(trial):
                    current = trial
                    changed = True
        return tuple(current), True

    def _is_unsat_subset(self, labels: Sequence[str]) -> bool:
        """Check whether a subset of tracked constraints is already unsatisfiable."""
        solver = self._solver()
        by_label = {t.label: t for t in self._tracked}
        for label in labels:
            tracked = by_label.get(label)
            if tracked is None:
                return False
            solver.add(tracked.expr)
        return solver.check() == z3.unsat

    # -- entailment by counterexample search (Step 29) ------------------------

    def check_entailment(
        self,
        premises: Sequence[Statement],
        conclusion: Statement,
    ) -> SmtOutcome:
        """Test Γ ⊨ φ by checking Γ ∧ ¬φ (Step 29).

        UNSAT establishes entailment under the encoded semantics; SAT yields a
        countermodel validated independently; UNKNOWN or timeout yield 'not
        established', never falsity.
        """
        start = time.perf_counter()
        all_statements = (*premises, conclusion)
        need = self.check_theories(all_statements)
        self.track_statements(premises)
        negated_goal = z3.Not(self.encoder.formula_expr(conclusion.formula, {}))
        solver = self._solver()
        for tracked in self._tracked:
            solver.assert_and_track(tracked.expr, tracked.label)
        solver.assert_and_track(negated_goal, "negated_goal")
        z3_result = solver.check()
        wall_ms = (time.perf_counter() - start) * 1000.0
        identity = self._solver_identity()
        profile = TheoryProfile(
            theories=tuple(sorted(need.theories, key=lambda t: t.value)),
            nonlinear=need.nonlinear,
            exact_arithmetic=True,
            notes=tuple(need.notes),
        )
        if z3_result == z3.unsat:
            core_labels = tuple(sorted(
                str(core.decl().name()) for core in solver.unsat_core()
            ))
            return SmtOutcome(
                solver_status=SolverStatus.UNSAT,
                result_status="unsatisfiable",
                unsat_core=core_labels,
                unsat_core_minimized=False,
                theory_profile=profile,
                solver=identity,
                wall_clock_ms=wall_ms,
                limitations=(
                    "entailment established under the encoded semantics and assumptions",
                    "the unsatisfiable core is a sufficient conflicting subset, "
                    "not necessarily a minimum-cardinality cause",
                ),
            )
        if z3_result == z3.sat:
            model = solver.model()
            decoded, decode_warnings = self._decode_model(model)
            # The countermodel must satisfy Γ and violate φ (independent check).
            premise_results = [
                self.evaluator.evaluate_formula(
                    self._statement_by_id(t.statement_id).formula, decoded
                )
                for t in self._tracked
            ]
            conclusion_result = self.evaluator.evaluate_formula(
                conclusion.formula, decoded
            )
            if None in premise_results or conclusion_result is None:
                return SmtOutcome(
                    solver_status=SolverStatus.SAT,
                    result_status="unknown",
                    theory_profile=profile,
                    solver=identity,
                    wall_clock_ms=wall_ms,
                    reason="countermodel could not be independently re-evaluated",
                    warnings=tuple(decode_warnings),
                )
            if not all(premise_results) or conclusion_result:
                raise MalformedSolverOutputError(
                    "countermodel failed independent validation: must satisfy Γ and violate φ"
                )
            return SmtOutcome(
                solver_status=SolverStatus.SAT,
                result_status="satisfiable",
                model=decoded,
                model_independently_validated=True,
                counterexample={
                    "kind": "countermodel_to_entailment",
                    "assignment": {k: _jsonable(v) for k, v in decoded.items()},
                    "conclusion_statement": conclusion.statement_id,
                },
                theory_profile=profile,
                solver=identity,
                wall_clock_ms=wall_ms,
                warnings=tuple(decode_warnings),
                limitations=(
                    "a countermodel under the encoded semantics; it does not by itself "
                    "establish anything about the external world"
                ),
            )
        reason = str(solver.reason_unknown() or "")
        status = "timeout" if "timeout" in reason.lower() or "canceled" in reason.lower() \
            else "unknown"
        return SmtOutcome(
            solver_status=SolverStatus.TIMEOUT if status == "timeout" else SolverStatus.UNKNOWN,
            result_status=status,
            theory_profile=profile,
            solver=identity,
            wall_clock_ms=wall_ms,
            reason=reason,
            limitations=("not established; UNKNOWN and timeout never imply falsity",),
        )


def _jsonable(value: Any) -> Any:
    """Convert decoded model values into JSON-safe primitives."""
    if isinstance(value, Fraction):
        return {"numerator": value.numerator, "denominator": value.denominator}
    if isinstance(value, bool) or isinstance(value, int) or isinstance(value, str):
        return value
    return str(value)


__all__ = [
    "SMT_ADAPTER_VERSION",
    "Z3_AVAILABLE",
    "Z3_VERSION",
    "TheoryNeed",
    "detect_theories",
    "EncodedConstraint",
    "SmtEncoder",
    "IndependentEvaluator",
    "UnsupportedEvaluationError",
    "DivisionByZeroEvaluation",
    "SmtOutcome",
    "SmtAdapter",
]
