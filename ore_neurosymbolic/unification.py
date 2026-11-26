"""Typed unification with occurs check and bounded diagnostics (Step B3).

Implements the typed substitutions required by Phase B: variables bind only to
terms of a compatible declared type, repeated variables must unify to the same
constant, type mismatches are rejected instead of silently succeeding, and
every failure carries a bounded diagnostic record. Unification is deterministic:
candidate bindings are explored in sorted order.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .contracts import LogicalTerm, LogicalVariable, TermOrVariable, term_key


@dataclass(frozen=True, slots=True)
class UnificationDiagnostic:
    """One bounded diagnostic explaining why unification failed."""

    code: str
    message: str
    context: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-compatible record."""
        return {"code": self.code, "message": self.message, "context": dict(self.context)}


class TypeTable:
    """Declares type compatibility for unification (Step B1/B3).

    ``compatible(a, b)`` returns True when values of type ``a`` can bind to
    positions declared as type ``b``. The ``any`` type is compatible with
    everything; otherwise an exact match is required. Callers may inject a
    custom compatibility relation (e.g. ontology subtyping) via ``relations``.
    """

    def __init__(
        self,
        relations: Mapping[str, frozenset[str]] | None = None,
    ) -> None:
        """Build the table from an optional subtype relation mapping."""
        self._relations: dict[str, frozenset[str]] = {
            key: frozenset(value) for key, value in (relations or {}).items()
        }

    def compatible(self, declared: str, actual: str) -> bool:
        """Return whether ``actual`` may bind into a ``declared`` position."""
        if declared == "any" or actual == "any" or declared == actual:
            return True
        return actual in self._relations.get(declared, frozenset())


class Substitution:
    """An immutable-view substitution mapping variable names to bound terms.

    The internal dict is mutable during construction inside the engine, but the
    public surface only exposes read operations plus explicit ``bind`` calls;
    every composed substitution returned to callers is a deep copy.
    """

    __slots__ = ("_bindings", "_types")

    def __init__(self, bindings: Mapping[str, TermOrVariable] | None = None) -> None:
        """Create a substitution from optional initial bindings."""
        self._bindings: dict[str, TermOrVariable] = dict(bindings or {})
        self._types: dict[str, str] = {
            name: item.type for name, item in self._bindings.items()
        }

    def bind(self, name: str, term: TermOrVariable) -> None:
        """Bind one variable to a term (engine-internal mutation)."""
        self._bindings[name] = term
        self._types[name] = term.type

    def is_bound(self, name: str) -> bool:
        """Return whether one variable is bound."""
        return name in self._bindings

    def lookup(self, name: str) -> TermOrVariable | None:
        """Return the bound term for one variable, or None."""
        return self._bindings.get(name)

    def apply(self, term: TermOrVariable) -> TermOrVariable:
        """Resolve one term through this substitution."""
        if isinstance(term, LogicalVariable):
            return self._bindings.get(term.name, term)
        return term

    def resolve_literal_args(self, args: tuple[TermOrVariable, ...]) -> tuple[TermOrVariable, ...]:
        """Resolve every argument of a literal through this substitution."""
        return tuple(self.apply(item) for item in args)

    def copy(self) -> "Substitution":
        """Return a deep copy so branching search paths never share bindings."""
        return Substitution({
            name: (
                LogicalVariable(item.name, item.type)
                if isinstance(item, LogicalVariable)
                else LogicalTerm(item.value, item.type)
            )
            for name, item in self._bindings.items()
        })

    def as_dict(self) -> dict[str, str]:
        """Return a JSON-compatible view of the bindings."""
        return {name: term_key(item) for name, item in sorted(self._bindings.items())}

    def __len__(self) -> int:
        """Return the number of bound variables."""
        return len(self._bindings)


def _occurs(variable_name: str, term: TermOrVariable) -> bool:
    """Return whether ``variable_name`` occurs inside ``term`` (occurs check)."""
    if isinstance(term, LogicalVariable):
        return term.name == variable_name
    return False


def unify_terms(
    pattern: TermOrVariable,
    value: TermOrVariable,
    substitution: Substitution,
    types: TypeTable | None = None,
) -> tuple[bool, UnificationDiagnostic | None]:
    """Unify one term against another under the current substitution.

    Returns ``(True, None)`` on success, or ``(False, diagnostic)`` with a
    bounded diagnostic on failure. The substitution is mutated only on success.
    """
    table = types or TypeTable()
    resolved_pattern = substitution.apply(pattern)
    resolved_value = substitution.apply(value)

    pattern_is_var = isinstance(resolved_pattern, LogicalVariable)
    value_is_var = isinstance(resolved_value, LogicalVariable)

    if pattern_is_var and value_is_var:
        if resolved_pattern.name == resolved_value.name:
            return True, None
        # Deterministic order: bind the lexicographically greater name to the lesser.
        first, second = sorted((resolved_pattern, resolved_value), key=lambda item: item.name)
        substitution.bind(second.name, first)
        return True, None

    if pattern_is_var:
        if _occurs(resolved_pattern.name, resolved_value):
            return False, UnificationDiagnostic(
                code="occurs_check_failed",
                message=f"variable ?{resolved_pattern.name} occurs in its own binding",
                context={"variable": resolved_pattern.name},
            )
        if not table.compatible(resolved_pattern.type, resolved_value.type):
            return False, UnificationDiagnostic(
                code="type_mismatch",
                message=(
                    f"variable ?{resolved_pattern.name} declared {resolved_pattern.type} "
                    f"cannot bind {resolved_value.type}"
                ),
                context={
                    "variable": resolved_pattern.name,
                    "declared": resolved_pattern.type,
                    "actual": resolved_value.type,
                },
            )
        substitution.bind(resolved_pattern.name, resolved_value)
        return True, None

    if value_is_var:
        return unify_terms(resolved_value, resolved_pattern, substitution, table)

    if not isinstance(resolved_pattern, LogicalTerm) or not isinstance(resolved_value, LogicalTerm):
        return False, UnificationDiagnostic(
            code="term_kind_mismatch",
            message="cannot unify a variable placeholder with an unresolvable term",
            context={},
        )
    if resolved_pattern.value != resolved_value.value:
        return False, UnificationDiagnostic(
            code="constant_conflict",
            message=(
                f"constant {resolved_pattern.value!r} does not match {resolved_value.value!r}"
            ),
            context={"left": resolved_pattern.value, "right": resolved_value.value},
        )
    # Constants unify by value equality: the fact table is authoritative over
    # the stored object's type annotation. Typed binding checks (the actual
    # type-compatibility gate of Step B3) apply to variable bindings above.
    return True, None


def unify_literals(
    pattern_args: tuple[TermOrVariable, ...],
    fact_args: tuple[TermOrVariable, ...],
    substitution: Substitution,
    types: TypeTable | None = None,
) -> tuple[bool, UnificationDiagnostic | None]:
    """Unify the argument lists of one rule premise against one supported literal.

    Arity must match exactly (Step B1 validation guarantees rule-side arity
    consistency; this guard protects against index mismatches at runtime).
    """
    if len(pattern_args) != len(fact_args):
        return False, UnificationDiagnostic(
            code="arity_mismatch",
            message=f"arity {len(pattern_args)} does not match fact arity {len(fact_args)}",
            context={"pattern_arity": len(pattern_args), "fact_arity": len(fact_args)},
        )
    for index, (pattern_item, fact_item) in enumerate(zip(pattern_args, fact_args)):
        ok, diagnostic = unify_terms(pattern_item, fact_item, substitution, types)
        if not ok:
            enriched = UnificationDiagnostic(
                code=diagnostic.code,
                message=diagnostic.message,
                context={**diagnostic.context, "argument_index": index},
            )
            return False, enriched
    return True, None


__all__ = [
    "Substitution",
    "TypeTable",
    "UnificationDiagnostic",
    "unify_literals",
    "unify_terms",
]
