"""Static policy checks on candidate genotypes (step 36).

Static checks run *before* any execution: parameter bounds against validated ranges,
action-vocabulary membership, policy-table bounds and, for policy IR, type checks,
depth/node limits, undefined feature references, unsafe actions, unreachable branches
(constant conditions) and resource bounds. Findings carry severities; errors make the
candidate ineligible for promotion.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from ..domain.actions import ACTION_VOCABULARY
from ..domain.genome import ParameterGenome, PolicyTableGenome
from ..search.genetic_programming import (
    BoolOpNode, CompareNode, IfActionNode, PolicyIRGenome, WhenActionNode,
    static_constant_value,
)

#: Finding severities.
SEVERITIES = ("error", "warning", "info")


@dataclass(frozen=True, slots=True)
class CheckFinding:
    """One static-check finding."""

    code: str
    severity: str
    message: str
    location: str = ""

    def __post_init__(self) -> None:
        if self.severity not in SEVERITIES:
            raise ValueError(f"unknown severity {self.severity!r}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code, "severity": self.severity,
            "message": self.message, "location": self.location,
        }


@dataclass(frozen=True, slots=True)
class StaticCheckReport:
    """Aggregate static-check result (step 36)."""

    findings: tuple[CheckFinding, ...] = ()
    complexity: Mapping[str, int] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """True when no *error*-severity finding was raised."""
        return not any(finding.severity == "error" for finding in self.findings)

    @property
    def errors(self) -> tuple[CheckFinding, ...]:
        return tuple(f for f in self.findings if f.severity == "error")

    @property
    def warnings(self) -> tuple[CheckFinding, ...]:
        return tuple(f for f in self.findings if f.severity == "warning")

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "findings": [finding.as_dict() for finding in self.findings],
            "complexity": dict(self.complexity),
        }


class StaticPolicyCheck:
    """Static validation of candidate genomes (step 36)."""

    def check(self, genome: Any) -> StaticCheckReport:
        findings: list[CheckFinding] = []
        complexity: dict[str, int] = {}
        schema_errors = genome.validate()
        for error in schema_errors:
            findings.append(CheckFinding(
                "schema-invalid", "error", error, location=genome.kind,
            ))
        capabilities = genome.capabilities()
        for action_name in sorted(capabilities):
            if action_name not in ACTION_VOCABULARY:
                findings.append(CheckFinding(
                    "unsafe-action", "error",
                    f"capability {action_name!r} is outside the approved vocabulary",
                    location=genome.kind,
                ))
        if genome.kind == "parameters":
            complexity["parameters"] = len(genome.values)
            findings.extend(self._check_parameters(genome))
        elif genome.kind == "policy_table":
            complexity["table_cells"] = len(genome.rules)
            findings.extend(self._check_policy_table(genome))
        elif genome.kind == "policy_ir":
            complexity["nodes"] = genome.root.size()
            complexity["depth"] = genome.root.depth()
            findings.extend(self._check_policy_ir(genome))
        else:
            findings.append(CheckFinding(
                "unknown-genome", "error",
                f"no static check for genome kind {genome.kind!r}",
            ))
        return StaticCheckReport(findings=tuple(findings), complexity=complexity)

    # ------------------------------------------------------------------ parameters
    def _check_parameters(self, genome: ParameterGenome) -> Iterable[CheckFinding]:
        findings: list[CheckFinding] = []
        for name in ("cpu_budget_share", "memory_budget_share"):
            value = genome.values.get(name)
            if value is not None and (value < 0.05 or value > 1.0):
                findings.append(CheckFinding(
                    "resource-bound", "warning",
                    f"{name}={value:.3f} is an unusual resource grant "
                    "(resource starvation safeguard)",
                ))
        route_total = sum(
            genome.values.get(f"route_weight__{queue}", 0.0)
            for queue in ("fast", "standard", "deep")
        )
        if route_total <= 0.0:
            findings.append(CheckFinding(
                "routing-degenerate", "error",
                "all routing weights are zero — no queue would receive tasks",
            ))
        return findings

    # ----------------------------------------------------------------- policy table
    def _check_policy_table(self, genome: PolicyTableGenome) -> Iterable[CheckFinding]:
        findings: list[CheckFinding] = []
        for key, actions in sorted(genome.rules.items()):
            if not actions:
                findings.append(CheckFinding(
                    "empty-cell", "warning",
                    f"policy cell {key!r} is empty (strategy abstains here)",
                    location="|".join(key),
                ))
            modules = [
                action.params.get("module") for action in actions
                if action.name == "activate_module"
            ]
            duplicates = len(modules) - len(set(modules))
            if duplicates:
                findings.append(CheckFinding(
                    "duplicate-action", "warning",
                    f"policy cell {key!r} repeats a module activation",
                    location="|".join(key),
                ))
        return findings

    # -------------------------------------------------------------------- policy IR
    def _check_policy_ir(self, genome: PolicyIRGenome) -> Iterable[CheckFinding]:
        findings: list[CheckFinding] = []
        root = genome.root
        if root.depth() > genome.grammar.max_depth:
            findings.append(CheckFinding(
                "depth-bound", "error",
                f"tree depth {root.depth()} exceeds the grammar bound "
                f"{genome.grammar.max_depth}",
            ))
        if root.size() > genome.grammar.max_nodes:
            findings.append(CheckFinding(
                "node-bound", "error",
                f"tree size {root.size()} exceeds the grammar bound "
                f"{genome.grammar.max_nodes} (GP bloat safeguard)",
            ))
        findings.extend(self._unreachable_branches(root))
        if root.size() > genome.grammar.max_nodes * 0.8:
            findings.append(CheckFinding(
                "bloat-pressure", "info",
                f"tree uses {root.size()}/{genome.grammar.max_nodes} nodes",
            ))
        return findings

    def _unreachable_branches(self, root: Any) -> Iterable[CheckFinding]:
        findings: list[CheckFinding] = []
        for node in _walk(root):
            if isinstance(node, (IfActionNode, WhenActionNode)):
                constant = static_constant_value(node.condition) if node.condition.ir_type == "num" else None
                if constant is None:
                    constant_bool = _constant_bool(node.condition)
                else:
                    constant_bool = None
                if constant_bool is True:
                    findings.append(CheckFinding(
                        "unreachable-branch", "warning",
                        "condition is statically constant-true: the alternative branch "
                        "is unreachable",
                    ))
                elif constant_bool is False:
                    findings.append(CheckFinding(
                        "unreachable-branch", "warning",
                        "condition is statically constant-false: the guarded action "
                        "never fires",
                    ))
        return findings


def _walk(node: Any) -> Iterable[Any]:
    yield node
    for child in node.children:
        yield from _walk(child)


def _constant_bool(node: Any) -> bool | None:
    """Statically evaluate a boolean node when fully constant, else None."""
    if getattr(node, "ir_type", None) != "bool":
        return None
    if isinstance(node, IfActionNode):
        return None
    if isinstance(node, CompareNode):
        nums = [static_constant_value(child) for child in node.children]
        if any(num is None for num in nums):
            return None
        return nums[0] < nums[1] if node.op == "lt" else nums[0] > nums[1]
    if isinstance(node, BoolOpNode):
        values = [_constant_bool(child) for child in node.children]
        if any(value is None for value in values):
            return None
        if node.op == "not":
            return not values[0]
        if node.op == "and":
            return all(values)
        if node.op == "or":
            return any(values)
    return None


__all__ = ["CheckFinding", "SEVERITIES", "StaticCheckReport", "StaticPolicyCheck"]
