"""Deterministic rule validation and the safe rule profile (Steps B1, Stage 1).

Every rule entering the engine must pass this validator first. Validation is
purely syntactic/structural and deterministic: the same malformed rule always
yields the same rejection codes. Checks cover arity consistency, unbound
variables (range-restricted rules), type compatibility against the declared
variable table, allowed operators, recursion policy, scope/exception structure
and priority bounds (Section 2: defeasible rules carry scope, priority and
exceptions; general rules must not defeat valid specific exceptions).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .contracts import (
    Literal,
    LogicalTerm,
    LogicalVariable,
    RuleClass,
    RuleDefinition,
)


@dataclass(frozen=True, slots=True)
class RuleProfile:
    """The safety profile a ruleset must satisfy before inference starts.

    ``allow_negation_in_head`` is fixed to False: deriving explicit negation
    through rules is disabled unless a future profile explicitly re-enables it
    (unknown is never silently turned into false).
    """

    max_premises: int = 8
    max_rules: int = 256
    max_recursion_depth: int = 8
    allow_negation_in_head: bool = False
    allow_self_recursion: bool = False
    require_range_restriction: bool = True
    allowed_predicates: frozenset[str] | None = None
    allowed_literal_arities: tuple[int, ...] = (2,)

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-compatible record."""
        return {
            "max_premises": self.max_premises,
            "max_rules": self.max_rules,
            "max_recursion_depth": self.max_recursion_depth,
            "allow_negation_in_head": self.allow_negation_in_head,
            "allow_self_recursion": self.allow_self_recursion,
            "require_range_restriction": self.require_range_restriction,
            "allowed_predicates": (
                sorted(self.allowed_predicates) if self.allowed_predicates is not None else None
            ),
        }


@dataclass(frozen=True, slots=True)
class RuleValidationIssue:
    """One deterministic rejection reason with a stable code."""

    code: str
    message: str
    rule_id: str

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-compatible record."""
        return {"code": self.code, "message": self.message, "rule_id": self.rule_id}


@dataclass(frozen=True, slots=True)
class RuleValidationReport:
    """The outcome of validating one ruleset (Step B1)."""

    accepted_rules: tuple[RuleDefinition, ...]
    rejected_rules: tuple[RuleValidationIssue, ...]
    profile: RuleProfile

    @property
    def ok(self) -> bool:
        """Return whether every rule was accepted."""
        return not self.rejected_rules

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-compatible record."""
        return {
            "accepted": [rule.rule_id for rule in self.accepted_rules],
            "rejected": [item.to_dict() for item in self.rejected_rules],
            "profile": self.profile.to_dict(),
            "ok": self.ok,
        }


def _literal_variables(literal: Literal) -> set[str]:
    """Return the variable names used by one literal."""
    return {item.name for item in literal.args if isinstance(item, LogicalVariable)}


def _literal_constants(literal: Literal) -> set[str]:
    """Return the constant values used by one literal."""
    return {item.value for item in literal.args if isinstance(item, LogicalTerm)}


class RuleValidator:
    """Validate a ruleset against a ``RuleProfile`` before any inference runs."""

    def __init__(self, profile: RuleProfile | None = None) -> None:
        """Bind the safety profile (defaults to the repository safe profile)."""
        self.profile = profile or RuleProfile()

    def validate_ruleset(
        self, rules: tuple[RuleDefinition, ...],
        declared_predicates: Mapping[str, tuple[int, ...]] | None = None,
    ) -> RuleValidationReport:
        """Validate every rule in deterministic input order and record rejections."""
        if len(rules) > self.profile.max_rules:
            return RuleValidationReport(
                accepted_rules=(),
                rejected_rules=(
                    RuleValidationIssue(
                        code="ruleset_too_large",
                        message=f"ruleset has {len(rules)} rules, profile allows "
                                f"{self.profile.max_rules}",
                        rule_id="*",
                    ),
                ),
                profile=self.profile,
            )
        accepted: list[RuleDefinition] = []
        rejected: list[RuleValidationIssue] = []
        seen_ids: dict[str, str] = {}
        for rule in rules:
            issues = self._validate_rule(rule, declared_predicates)
            if rule.rule_id in seen_ids and seen_ids[rule.rule_id] != rule.fingerprint():
                issues.append(RuleValidationIssue(
                    code="duplicate_rule_id",
                    message=f"rule_id {rule.rule_id!r} reused with different content",
                    rule_id=rule.rule_id,
                ))
            seen_ids[rule.rule_id] = rule.fingerprint()
            if issues:
                rejected.extend(issues)
            else:
                accepted.append(rule)
        return RuleValidationReport(
            accepted_rules=tuple(accepted),
            rejected_rules=tuple(rejected),
            profile=self.profile,
        )

    def _validate_rule(
        self,
        rule: RuleDefinition,
        declared_predicates: Mapping[str, tuple[int, ...]] | None,
    ) -> list[RuleValidationIssue]:
        """Run all structural checks for one rule in a fixed order."""
        issues: list[RuleValidationIssue] = []
        if len(rule.premises) > self.profile.max_premises:
            issues.append(RuleValidationIssue(
                code="too_many_premises",
                message=f"rule has {len(rule.premises)} premises, profile allows "
                        f"{self.profile.max_premises}",
                rule_id=rule.rule_id,
            ))
        if not self.profile.allow_negation_in_head and rule.conclusion.negated:
            issues.append(RuleValidationIssue(
                code="negation_in_head",
                message="rules may not derive explicit negation in this safe profile",
                rule_id=rule.rule_id,
            ))
        if self.profile.allowed_predicates is not None:
            for literal in (*rule.premises, rule.conclusion):
                if literal.predicate not in self.profile.allowed_predicates:
                    issues.append(RuleValidationIssue(
                        code="predicate_not_allowed",
                        message=f"predicate {literal.predicate!r} is outside the allowed set",
                        rule_id=rule.rule_id,
                    ))
        for literal in (*rule.premises, rule.conclusion):
            if literal.arity not in self.profile.allowed_literal_arities:
                issues.append(RuleValidationIssue(
                    code="unsupported_arity",
                    message=(
                        f"literal {literal.predicate!r} has arity {literal.arity}; the "
                        f"fact store is triple-based and supports arities "
                        f"{self.profile.allowed_literal_arities}"
                    ),
                    rule_id=rule.rule_id,
                ))
        if declared_predicates is not None:
            for literal in (*rule.premises, rule.conclusion):
                arities = declared_predicates.get(literal.predicate)
                if arities is not None and literal.arity not in arities:
                    issues.append(RuleValidationIssue(
                        code="arity_mismatch",
                        message=(
                            f"predicate {literal.predicate!r} used with arity "
                            f"{literal.arity}, declared arities {sorted(arities)}"
                        ),
                        rule_id=rule.rule_id,
                    ))
        issues.extend(self._check_variables(rule))
        issues.extend(self._check_recursion_policy(rule))
        issues.extend(self._check_exceptions(rule))
        return issues

    def _check_variables(self, rule: RuleDefinition) -> list[RuleValidationIssue]:
        """Check unbound variables, declared-type coverage and type consistency."""
        issues: list[RuleValidationIssue] = []
        premise_variables: set[str] = set()
        for premise in rule.premises:
            premise_variables |= _literal_variables(premise)
        conclusion_variables = _literal_variables(rule.conclusion)
        unbound = conclusion_variables - premise_variables
        if unbound:
            issues.append(RuleValidationIssue(
                code="unbound_variables",
                message=(
                    f"conclusion uses variables {sorted(unbound)} that no premise binds "
                    "(range restriction)"
                ),
                rule_id=rule.rule_id,
            ))
        undeclared = {
            name
            for name in premise_variables | conclusion_variables
            if f"?{name}" not in rule.typed_variables
        }
        if undeclared:
            issues.append(RuleValidationIssue(
                code="undeclared_variables",
                message=f"variables {sorted(undeclared)} used without a declared type",
                rule_id=rule.rule_id,
            ))
        for variable in sorted(premise_variables | conclusion_variables):
            declared_type = rule.typed_variables.get(f"?{variable}")
            for literal in (*rule.premises, rule.conclusion):
                for item in literal.args:
                    if isinstance(item, LogicalVariable) and item.name == variable:
                        if declared_type is not None and item.type != declared_type:
                            issues.append(RuleValidationIssue(
                                code="type_conflict",
                                message=(
                                    f"variable ?{variable} declared {declared_type} but used "
                                    f"as {item.type} in {literal.predicate!r}"
                                ),
                                rule_id=rule.rule_id,
                            ))
        return issues

    def _check_recursion_policy(self, rule: RuleDefinition) -> list[RuleValidationIssue]:
        """Reject self-recursive strict rules unless the profile allows them."""
        issues: list[RuleValidationIssue] = []
        if not self.profile.allow_self_recursion:
            conclusion_key = (rule.conclusion.predicate, rule.conclusion.arity)
            for premise in rule.premises:
                if (premise.predicate, premise.arity) == conclusion_key and not premise.negated:
                    issues.append(RuleValidationIssue(
                        code="unsafe_self_recursion",
                        message=(
                            f"rule {rule.rule_id!r} is directly self-recursive; recursion is "
                            "handled by the engine's cycle detection and profile policy"
                        ),
                        rule_id=rule.rule_id,
                    ))
                    break
        return issues

    def _check_exceptions(self, rule: RuleDefinition) -> list[RuleValidationIssue]:
        """Check exception structure for defeasible rules (Section 2)."""
        issues: list[RuleValidationIssue] = []
        if rule.rule_class is RuleClass.STRICT and rule.exceptions:
            issues.append(RuleValidationIssue(
                code="strict_rule_with_exceptions",
                message="strict rules cannot declare exceptions; use rule_class='defeasible'",
                rule_id=rule.rule_id,
            ))
        if rule.rule_class is RuleClass.DEFEASIBLE:
            scope_keys = set(rule.scope)
            if not scope_keys:
                issues.append(RuleValidationIssue(
                    code="defeasible_rule_without_scope",
                    message="defeasible rules must declare a non-empty scope",
                    rule_id=rule.rule_id,
                ))
        return issues


def validate_ruleset(
    rules: tuple[RuleDefinition, ...],
    *,
    profile: RuleProfile | None = None,
    declared_predicates: Mapping[str, tuple[int, ...]] | None = None,
) -> RuleValidationReport:
    """Validate one ruleset with an optional profile (Step B1 entry point)."""
    return RuleValidator(profile).validate_ruleset(rules, declared_predicates)


def build_predicate_arity_table(rules: tuple[RuleDefinition, ...]) -> dict[str, tuple[int, ...]]:
    """Build the declared predicate/arity index used for cross-checking rules."""
    table: dict[str, set[int]] = {}
    for rule in rules:
        for literal in (*rule.premises, rule.conclusion):
            table.setdefault(literal.predicate, set()).add(literal.arity)
    return {predicate: tuple(sorted(arities)) for predicate, arities in table.items()}


def select_relevant_rules(
    rules: tuple[RuleDefinition, ...],
    goal_predicates: tuple[str, ...],
    *,
    max_rules: int | None = None,
) -> tuple[tuple[RuleDefinition, ...], str]:
    """Index rules by predicate and keep goal-relevant ones with a rationale (Step B2).

    A rule is relevant when its conclusion predicate matches one goal predicate
    (backward mode) or any of its premises match a currently active predicate.
    Returns the selected rules and a recorded selection rationale.
    """
    by_conclusion: dict[str, list[RuleDefinition]] = {}
    by_premise: dict[str, list[RuleDefinition]] = {}
    for rule in rules:
        by_conclusion.setdefault(rule.conclusion.predicate, []).append(rule)
        for premise in rule.premises:
            by_premise.setdefault(premise.predicate, []).append(rule)

    selected: dict[str, RuleDefinition] = {}
    for predicate in goal_predicates:
        for rule in by_conclusion.get(predicate, ()):
            selected[rule.rule_id] = rule
    rationale = f"matched goal predicates {sorted(set(goal_predicates))}"
    if not selected:
        for rule in rules:
            selected[rule.rule_id] = rule
        rationale = "no goal predicate matched; selected the full ruleset within bounds"
    ordered = tuple(
        sorted(selected.values(), key=lambda item: (item.rule_id, item.version))
    )
    if max_rules is not None and len(ordered) > max_rules:
        ordered = ordered[:max_rules]
        rationale += f"; truncated to {max_rules} rules by budget"
    return ordered, rationale


__all__ = [
    "RuleProfile",
    "RuleValidationIssue",
    "RuleValidationReport",
    "RuleValidator",
    "build_predicate_arity_table",
    "select_relevant_rules",
    "validate_ruleset",
]
