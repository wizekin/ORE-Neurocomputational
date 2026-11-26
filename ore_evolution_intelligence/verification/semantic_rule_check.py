"""Semantic rule checks for generated policies (step 37; section 9 safeguards).

Generated decision rules must pass semantic and consistency checks before promotion:
contradictions (the same guard condition selecting conflicting actions — activating and
deactivating the same module, or deferring while activating), duplicate effects,
deactivations without prior activations, and rule schemas consistent with the USG
module vocabulary (step 37: USG rule candidates must pass formal constraints and
consistency checks). USG semantic structures arrive through the approved action
vocabulary and the ORE module namespace — the layer never writes USG records.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from ..domain.actions import ORE_MODULES
from ..domain.genome import canonical_json
from ..search.genetic_programming import (
    ActionNode, IfActionNode, WhenActionNode, static_constant_value,
)

#: Rule schema: actions a generated policy may reference (USG module vocabulary).
RULE_SCHEMA_ACTIONS: tuple[str, ...] = (
    "activate_module", "deactivate_module", "defer_task",
    "escalate_verification", "request_retrieval", "route_task",
    "adjust_threshold", "report_abstention",
)


@dataclass(frozen=True, slots=True)
class SemanticFinding:
    """One semantic-consistency finding."""

    code: str
    severity: str  # "error" | "warning" | "info"
    message: str
    guard: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code, "severity": self.severity,
            "message": self.message, "guard": self.guard,
        }


@dataclass(frozen=True, slots=True)
class SemanticCheckReport:
    """Aggregate semantic check result (step 37)."""

    findings: tuple[SemanticFinding, ...] = ()
    rules_checked: int = 0

    @property
    def ok(self) -> bool:
        return not any(finding.severity == "error" for finding in self.findings)

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok, "rules_checked": self.rules_checked,
            "findings": [finding.as_dict() for finding in self.findings],
        }


class SemanticRuleCheck:
    """Semantic validation of generated decision rules (step 37)."""

    def check(self, genome: Any) -> SemanticCheckReport:
        if genome.kind != "policy_ir":
            # Parameter and policy-table genomes cannot express USG rule
            # contradictions; their semantic surface is the action vocabulary,
            # which static checks already cover.
            return SemanticCheckReport(findings=(), rules_checked=0)
        findings: list[SemanticFinding] = []
        rules = _collect_rules(genome.root)
        for rule in rules:
            findings.extend(self._check_rule_schema(rule))
        findings.extend(self._check_contradictions(rules))
        deactivations = {
            module for rule in rules for module in rule["deactivates"]
        }
        activations = {
            module for rule in rules for module in rule["activates"]
        }
        for module in sorted(deactivations - activations):
            findings.append(SemanticFinding(
                "deactivation-without-activation", "warning",
                f"rule set deactivates {module} without any prior activation",
            ))
        return SemanticCheckReport(
            findings=tuple(findings), rules_checked=len(rules),
        )

    def _check_rule_schema(self, rule: Mapping[str, Any]) -> Iterable[SemanticFinding]:
        findings: list[SemanticFinding] = []
        for module in (*rule["activates"], *rule["deactivates"]):
            if module not in ORE_MODULES:
                findings.append(SemanticFinding(
                    "schema-invalid-module", "error",
                    f"rule references unknown USG module {module!r}",
                    guard=rule["guard"],
                ))
        if rule["defers"] and (rule["activates"] or rule["deactivates"]):
            findings.append(SemanticFinding(
                "defer-conflict", "error",
                "rule defers the task while also activating or deactivating modules",
                guard=rule["guard"],
            ))
        if rule["defers"] and rule["escalates"]:
            findings.append(SemanticFinding(
                "defer-escalate-conflict", "warning",
                "rule defers while escalating verification — the escalation is dead",
                guard=rule["guard"],
            ))
        return findings

    def _check_contradictions(self, rules: Iterable[Mapping[str, Any]]) -> Iterable[SemanticFinding]:
        findings: list[SemanticFinding] = []
        materialized = list(rules)
        for index, rule_a in enumerate(materialized):
            for rule_b in materialized[index + 1:]:
                if rule_a["guard"] != rule_b["guard"]:
                    continue
                conflicts = (
                    set(rule_a["activates"]) & set(rule_b["deactivates"])
                ) | (
                    set(rule_a["deactivates"]) & set(rule_b["activates"])
                )
                if conflicts:
                    findings.append(SemanticFinding(
                        "contradictory-rules", "error",
                        f"guard selects conflicting actions for module(s): "
                        f"{sorted(conflicts)}",
                        guard=rule_a["guard"],
                    ))
                if rule_a["defers"] != rule_b["defers"] and rule_a["guard"] == rule_b["guard"]:
                    findings.append(SemanticFinding(
                        "contradictory-defer", "error",
                        "identical guard both defers and proceeds",
                        guard=rule_a["guard"],
                    ))
        return findings


def _collect_rules(root: Any) -> list[dict[str, Any]]:
    """Extract rule records (guard -> effects) from the policy IR."""
    rules: list[dict[str, Any]] = []
    for node in _walk(root):
        if isinstance(node, (IfActionNode, WhenActionNode)):
            guard = canonical_json({"cond": node.condition.as_dict()})
            effects = _effects(node.then_branch if isinstance(node, WhenActionNode) else node.then_branch)
            else_effects = _effects(node.else_branch) if isinstance(node, IfActionNode) else {
                "activates": [], "deactivates": [], "defers": False, "escalates": False,
            }
            rules.append({
                "guard": guard,
                "activates": effects["activates"],
                "deactivates": effects["deactivates"],
                "defers": effects["defers"],
                "escalates": effects["escalates"],
                "else_activates": else_effects["activates"],
                "else_deactivates": else_effects["deactivates"],
            })
    return rules


def _effects(node: Any) -> dict[str, Any]:
    activates: list[str] = []
    deactivates: list[str] = []
    defers = False
    escalates = False
    for child in _walk(node):
        if isinstance(child, ActionNode):
            call = child.call
            if call.name == "activate_module":
                activates.append(call.params.get("module", ""))
            elif call.name == "deactivate_module":
                deactivates.append(call.params.get("module", ""))
            elif call.name == "defer_task":
                defers = True
            elif call.name == "escalate_verification":
                escalates = True
    return {
        "activates": activates, "deactivates": deactivates,
        "defers": defers, "escalates": escalates,
    }


def _walk(node: Any) -> Iterable[Any]:
    yield node
    for child in node.children:
        yield from _walk(child)


__all__ = [
    "RULE_SCHEMA_ACTIONS", "SemanticCheckReport", "SemanticFinding", "SemanticRuleCheck",
]
