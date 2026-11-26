"""Constrained genetic programming over a typed grammar (Phase D, steps 31-38).

Genotypes are *typed expression trees* (step 31) built from approved terminals (task
features, resource indicators, constants) and approved operators (comparisons, guarded
arithmetic, boolean logic, action selectors). Trees are bounded at construction
(step 32: maximum depth, node count, type compatibility, allowed functions) and
*compiled to a declarative policy IR* (step 33): an inspectable, serializable tree
interpreted by a bounded evaluator — never generated Python source.

Subtree crossover and mutation exchange compatible typed subtrees and revalidate
offspring (step 34); anything that breaks depth, size, type or action-vocabulary
bounds is rejected. Complexity is tracked (step 35); static and semantic checks live
in :mod:`..verification` (steps 36-37). Generated programs stay in the candidate
registry lifecycle — they cannot modify USG or production coordinator behavior
directly (step 38).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from ..domain.actions import ORE_MODULES, ActionCall, validate_action

#: IR value types (step 31: typed grammar).
IR_TYPES = ("num", "bool", "action")

#: Approved numeric operators with arity (step 31).
NUM_OPS: dict[str, int] = {
    "add": 2, "sub": 2, "mul": 2, "div": 2, "min": 2, "max": 2, "clamp01": 1,
}
#: Approved boolean operators with arity.
BOOL_OPS: dict[str, int] = {"and": 2, "or": 2, "not": 1}
#: Approved comparison operators with arity.
COMPARE_OPS: dict[str, int] = {"lt": 2, "gt": 2}
#: Approved action combinators: (arity, argument types).
ACTION_OPS: dict[str, tuple[int, tuple[str, ...]]] = {
    "if": (3, ("bool", "action", "action")),
    "when": (2, ("bool", "action")),
    "combine": (2, ("action", "action")),
}

#: Task-feature terminals exposed to generated policies (step 31). The family
#: index lets typed policies branch per task family (0..5 in family order).
POLICY_FEATURES: tuple[tuple[str, str], ...] = (
    ("complexity", "estimated task complexity in [0,1]"),
    ("novelty", "task novelty estimate in [0,1]"),
    ("latency_sensitivity", "latency sensitivity in [0,1]"),
    ("resource_pressure", "resource pressure indicator in {0,1}"),
    ("safety_critical", "safety-critical flag in {0,1}"),
    ("task_family_index", "task family index 0..5"),
)

#: Approved numeric constants (step 31: approved terminals). The 1.5..4.5 family
#: thresholds are the grammar designer's approved breakpoints for task-family
#: branching (``task_family_index`` in 0..5).
POLICY_CONSTANTS: tuple[float, ...] = (0.25, 0.5, 0.75, 1.5, 2.5, 3.5, 4.5)

GRAMMAR_VERSION = "policy-grammar-v1"


class _StepBudget:
    """Bounded-evaluation step counter (steps 12, 16)."""

    __slots__ = ("limit", "used")

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.used = 0

    def tick(self) -> None:
        self.used += 1
        if self.used > self.limit:
            raise _StepBudgetExceeded()


class _StepBudgetExceeded(Exception):
    pass


def _finite(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return float(min(max(value, -1e6), 1e6))


class IRNode:
    """Base class for typed policy-IR nodes (step 33: declarative, inspectable)."""

    @property
    def ir_type(self) -> str:
        raise NotImplementedError

    @property
    def children(self) -> tuple["IRNode", ...]:
        return ()

    def depth(self) -> int:
        child_depths = [child.depth() for child in self.children]
        return 1 + (max(child_depths) if child_depths else 0)

    def size(self) -> int:
        return 1 + sum(child.size() for child in self.children)

    def as_dict(self) -> dict[str, Any]:
        raise NotImplementedError

    def evaluate(self, context: Mapping[str, float], budget: _StepBudget) -> Any:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class FeatureNode(IRNode):
    name: str

    @property
    def ir_type(self) -> str:
        return "num"

    def as_dict(self) -> dict[str, Any]:
        return {"t": "feature", "name": self.name}

    def evaluate(self, context: Mapping[str, float], budget: _StepBudget) -> float:
        budget.tick()
        if self.name not in context:
            raise KeyError(f"undefined feature reference: {self.name!r}")
        return _finite(float(context[self.name]))


@dataclass(frozen=True, slots=True)
class ConstantNode(IRNode):
    value: float

    @property
    def ir_type(self) -> str:
        return "num"

    def as_dict(self) -> dict[str, Any]:
        return {"t": "const", "value": self.value}

    def evaluate(self, context: Mapping[str, float], budget: _StepBudget) -> float:
        budget.tick()
        return _finite(self.value)


@dataclass(frozen=True, slots=True)
class NumOpNode(IRNode):
    op: str
    operands: tuple[IRNode, ...]

    @property
    def ir_type(self) -> str:
        return "num"

    @property
    def children(self) -> tuple[IRNode, ...]:
        return self.operands

    def as_dict(self) -> dict[str, Any]:
        return {"t": "numop", "op": self.op, "children": [c.as_dict() for c in self.operands]}

    def evaluate(self, context: Mapping[str, float], budget: _StepBudget) -> float:
        budget.tick()
        values = [child.evaluate(context, budget) for child in self.operands]
        if self.op == "add":
            return _finite(values[0] + values[1])
        if self.op == "sub":
            return _finite(values[0] - values[1])
        if self.op == "mul":
            return _finite(values[0] * values[1])
        if self.op == "div":
            return _finite(values[0] / values[1]) if values[1] != 0.0 else 1.0
        if self.op == "min":
            return _finite(min(values))
        if self.op == "max":
            return _finite(max(values))
        if self.op == "clamp01":
            return _finite(min(max(values[0], 0.0), 1.0))
        raise ValueError(f"operator {self.op!r} is not an approved numeric operator")


@dataclass(frozen=True, slots=True)
class CompareNode(IRNode):
    op: str
    operands: tuple[IRNode, ...]

    @property
    def ir_type(self) -> str:
        return "bool"

    @property
    def children(self) -> tuple[IRNode, ...]:
        return self.operands

    def as_dict(self) -> dict[str, Any]:
        return {"t": "cmp", "op": self.op, "children": [c.as_dict() for c in self.operands]}

    def evaluate(self, context: Mapping[str, float], budget: _StepBudget) -> bool:
        budget.tick()
        left = self.operands[0].evaluate(context, budget)
        right = self.operands[1].evaluate(context, budget)
        if self.op == "lt":
            return left < right
        if self.op == "gt":
            return left > right
        raise ValueError(f"operator {self.op!r} is not an approved comparison")


@dataclass(frozen=True, slots=True)
class BoolOpNode(IRNode):
    op: str
    operands: tuple[IRNode, ...]

    @property
    def ir_type(self) -> str:
        return "bool"

    @property
    def children(self) -> tuple[IRNode, ...]:
        return self.operands

    def as_dict(self) -> dict[str, Any]:
        return {"t": "boolop", "op": self.op, "children": [c.as_dict() for c in self.operands]}

    def evaluate(self, context: Mapping[str, float], budget: _StepBudget) -> bool:
        budget.tick()
        if self.op == "not":
            return not self.operands[0].evaluate(context, budget)
        left = self.operands[0].evaluate(context, budget)
        if self.op == "and":
            return left and self.operands[1].evaluate(context, budget)
        if self.op == "or":
            return left or self.operands[1].evaluate(context, budget)
        raise ValueError(f"operator {self.op!r} is not an approved boolean operator")


@dataclass(frozen=True, slots=True)
class ActionNode(IRNode):
    call: ActionCall

    @property
    def ir_type(self) -> str:
        return "action"

    def as_dict(self) -> dict[str, Any]:
        return {"t": "action", "call": self.call.as_dict()}

    def evaluate(self, context: Mapping[str, float], budget: _StepBudget) -> frozenset[ActionCall]:
        budget.tick()
        return frozenset({self.call})


@dataclass(frozen=True, slots=True)
class IfActionNode(IRNode):
    condition: IRNode
    then_branch: IRNode
    else_branch: IRNode

    @property
    def ir_type(self) -> str:
        return "action"

    @property
    def children(self) -> tuple[IRNode, ...]:
        return (self.condition, self.then_branch, self.else_branch)

    def as_dict(self) -> dict[str, Any]:
        return {
            "t": "if", "cond": self.condition.as_dict(),
            "then": self.then_branch.as_dict(), "else": self.else_branch.as_dict(),
        }

    def evaluate(self, context: Mapping[str, float], budget: _StepBudget) -> frozenset[ActionCall]:
        budget.tick()
        if self.condition.evaluate(context, budget):
            return self.then_branch.evaluate(context, budget)
        return self.else_branch.evaluate(context, budget)


@dataclass(frozen=True, slots=True)
class WhenActionNode(IRNode):
    condition: IRNode
    then_branch: IRNode

    @property
    def ir_type(self) -> str:
        return "action"

    @property
    def children(self) -> tuple[IRNode, ...]:
        return (self.condition, self.then_branch)

    def as_dict(self) -> dict[str, Any]:
        return {
            "t": "when", "cond": self.condition.as_dict(),
            "then": self.then_branch.as_dict(),
        }

    def evaluate(self, context: Mapping[str, float], budget: _StepBudget) -> frozenset[ActionCall]:
        budget.tick()
        if self.condition.evaluate(context, budget):
            return self.then_branch.evaluate(context, budget)
        return frozenset()


@dataclass(frozen=True, slots=True)
class CombineActionNode(IRNode):
    left: IRNode
    right: IRNode

    @property
    def ir_type(self) -> str:
        return "action"

    @property
    def children(self) -> tuple[IRNode, ...]:
        return (self.left, self.right)

    def as_dict(self) -> dict[str, Any]:
        return {"t": "combine", "a": self.left.as_dict(), "b": self.right.as_dict()}

    def evaluate(self, context: Mapping[str, float], budget: _StepBudget) -> frozenset[ActionCall]:
        budget.tick()
        left = self.left.evaluate(context, budget)
        right = self.right.evaluate(context, budget)
        return frozenset(left | right)


def node_from_dict(data: Mapping[str, Any]) -> IRNode:
    """Rebuild one node from its serialized form (schema-valid by construction)."""
    tag = data.get("t")
    if tag == "feature":
        return FeatureNode(data["name"])
    if tag == "const":
        return ConstantNode(float(data["value"]))
    if tag == "numop":
        return NumOpNode(data["op"], tuple(node_from_dict(c) for c in data["children"]))
    if tag == "cmp":
        return CompareNode(data["op"], tuple(node_from_dict(c) for c in data["children"]))
    if tag == "boolop":
        return BoolOpNode(data["op"], tuple(node_from_dict(c) for c in data["children"]))
    if tag == "action":
        return ActionNode(ActionCall(data["call"]["name"], data["call"].get("params", {})))
    if tag == "if":
        return IfActionNode(
            node_from_dict(data["cond"]), node_from_dict(data["then"]),
            node_from_dict(data["else"]),
        )
    if tag == "when":
        return WhenActionNode(node_from_dict(data["cond"]), node_from_dict(data["then"]))
    if tag == "combine":
        return CombineActionNode(node_from_dict(data["a"]), node_from_dict(data["b"]))
    raise ValueError(f"unknown policy-IR node tag {tag!r}")


class TypedGrammar:
    """Typed grammar with bounded construction (steps 31-32).

    Construction enforces maximum depth, node count, type compatibility and allowed
    functions; ``validate`` re-checks every constraint for deserialized trees.
    """

    def __init__(
        self,
        *,
        features: Sequence[tuple[str, str]] = POLICY_FEATURES,
        constants: Sequence[float] = POLICY_CONSTANTS,
        modules: Sequence[str] = ORE_MODULES,
        max_depth: int = 8,
        max_nodes: int = 64,
    ) -> None:
        if max_depth < 2 or max_nodes < 4:
            raise ValueError("grammar bounds are too small")
        self.features = tuple(features)
        self.constants = tuple(constants)
        self.modules = tuple(modules)
        self.max_depth = max_depth
        self.max_nodes = max_nodes
        self.version = GRAMMAR_VERSION

    # ------------------------------------------------------------------ validation
    def validate(self, root: IRNode) -> list[str]:
        """Full structural validation: types, bounds, allowlists (step 32, 36)."""
        errors: list[str] = []
        if root.ir_type != "action":
            errors.append(f"policy root must have type 'action', got {root.ir_type!r}")
        if root.depth() > self.max_depth:
            errors.append(
                f"tree depth {root.depth()} exceeds bound {self.max_depth}"
            )
        if root.size() > self.max_nodes:
            errors.append(
                f"tree size {root.size()} exceeds bound {self.max_nodes}"
            )
        errors.extend(self._validate_node(root))
        return errors

    def _validate_node(self, node: IRNode) -> list[str]:
        errors: list[str] = []
        if isinstance(node, FeatureNode):
            if node.name not in {name for name, _doc in self.features}:
                errors.append(f"unknown feature terminal {node.name!r}")
        elif isinstance(node, ConstantNode):
            if node.value not in self.constants:
                errors.append(f"constant {node.value!r} is not an approved terminal")
        elif isinstance(node, NumOpNode):
            if node.op not in NUM_OPS or len(node.operands) != NUM_OPS[node.op]:
                errors.append(f"numeric operator {node.op!r} violates the grammar")
            for operand in node.operands:
                if operand.ir_type != "num":
                    errors.append(
                        f"numeric operator {node.op!r} needs 'num' operands, got {operand.ir_type!r}"
                    )
        elif isinstance(node, CompareNode):
            if node.op not in COMPARE_OPS or len(node.operands) != 2:
                errors.append(f"comparison {node.op!r} violates the grammar")
            for operand in node.operands:
                if operand.ir_type != "num":
                    errors.append(
                        f"comparison {node.op!r} needs 'num' operands, got {operand.ir_type!r}"
                    )
        elif isinstance(node, BoolOpNode):
            if node.op not in BOOL_OPS or len(node.operands) != BOOL_OPS[node.op]:
                errors.append(f"boolean operator {node.op!r} violates the grammar")
            for operand in node.operands:
                if operand.ir_type != "bool":
                    errors.append(
                        f"boolean operator {node.op!r} needs 'bool' operands, got {operand.ir_type!r}"
                    )
        elif isinstance(node, ActionNode):
            problems = validate_action(node.call.name, node.call.params)
            if problems:
                errors.append(f"action terminal violates the vocabulary: {problems[0]}")
        elif isinstance(node, IfActionNode):
            if node.condition.ir_type != "bool":
                errors.append("'if' combinator needs a 'bool' condition")
            if node.then_branch.ir_type != "action" or node.else_branch.ir_type != "action":
                errors.append("'if' combinator needs 'action' branches")
        elif isinstance(node, WhenActionNode):
            if node.condition.ir_type != "bool":
                errors.append("'when' combinator needs a 'bool' condition")
            if node.then_branch.ir_type != "action":
                errors.append("'when' combinator needs an 'action' branch")
        elif isinstance(node, CombineActionNode):
            if node.left.ir_type != "action" or node.right.ir_type != "action":
                errors.append("'combine' combinator needs 'action' operands")
        else:
            errors.append(f"unknown node type {type(node).__name__!r}")
        for child in node.children:
            errors.extend(self._validate_node(child))
        return errors

    # ----------------------------------------------------------------- generation
    def random_tree(
        self,
        rng: Any,
        *,
        ir_type: str = "action",
        depth_limit: int | None = None,
    ) -> IRNode:
        """Type-directed random tree within bounds (step 32)."""
        limit = depth_limit if depth_limit is not None else self.max_depth
        for _ in range(20):
            tree = self._grow(rng, ir_type, limit)
            if self.validate(tree) == []:
                return tree
        # Guaranteed fallback: a single bounded terminal.
        return self._terminal(rng, ir_type)

    def _grow(self, rng: Any, ir_type: str, depth_left: int) -> IRNode:
        if depth_left <= 1 or rng.uniform() < 0.35:
            return self._terminal(rng, ir_type)
        if ir_type == "num":
            op = rng.choice(sorted(NUM_OPS))
            operands = tuple(self._grow(rng, "num", depth_left - 1) for _ in range(NUM_OPS[op]))
            return NumOpNode(op, operands)
        if ir_type == "bool":
            roll = rng.uniform()
            if roll < 0.6:
                op = rng.choice(sorted(COMPARE_OPS))
                operands = (
                    self._grow(rng, "num", depth_left - 1),
                    self._grow(rng, "num", depth_left - 1),
                )
                return CompareNode(op, operands)
            op = rng.choice(sorted(BOOL_OPS))
            operands = tuple(
                self._grow(rng, "bool", depth_left - 1) for _ in range(BOOL_OPS[op])
            )
            return BoolOpNode(op, operands)
        # action
        roll = rng.uniform()
        if roll < 0.4:
            condition = self._grow(rng, "bool", depth_left - 1)
            return IfActionNode(
                condition,
                self._grow(rng, "action", depth_left - 1),
                self._grow(rng, "action", depth_left - 1),
            )
        if roll < 0.7:
            condition = self._grow(rng, "bool", depth_left - 1)
            return WhenActionNode(condition, self._grow(rng, "action", depth_left - 1))
        return CombineActionNode(
            self._grow(rng, "action", depth_left - 1),
            self._grow(rng, "action", depth_left - 1),
        )

    def _terminal(self, rng: Any, ir_type: str) -> IRNode:
        if ir_type == "num":
            if rng.uniform() < 0.5 and self.features:
                name, _doc = rng.choice(list(self.features))
                return FeatureNode(name)
            return ConstantNode(rng.choice(list(self.constants)))
        if ir_type == "bool":
            left = self._terminal(rng, "num")
            right = self._terminal(rng, "num")
            return CompareNode(rng.choice(("lt", "gt")), (left, right))
        # action terminal: activate a module, defer, or escalate
        roll = rng.uniform()
        if roll < 0.75:
            module = rng.choice(list(self.modules))
            return ActionNode(ActionCall("activate_module", {"module": module}))
        if roll < 0.9:
            return ActionNode(ActionCall("defer_task", {"reason": "low_confidence"}))
        return ActionNode(ActionCall("escalate_verification", {}))


DEFAULT_GRAMMAR = TypedGrammar()


def random_tree(grammar: TypedGrammar, rng: Any, *, max_depth: int | None = None) -> IRNode:
    """Random bounded tree rooted at the policy type (step 32)."""
    return grammar.random_tree(rng, depth_limit=max_depth)


def _collect_nodes(node: IRNode, path: tuple[int, ...] = ()) -> list[tuple[tuple[int, ...], IRNode]]:
    entries: list[tuple[tuple[int, ...], IRNode]] = [((path), node)]
    for index, child in enumerate(node.children):
        entries.extend(_collect_nodes(child, (*path, index)))
    return entries


def _replace_path(node: IRNode, path: Sequence[int], subtree: IRNode) -> IRNode:
    """Rebuild the tree with ``subtree`` at ``path`` (path from the root)."""
    if not path:
        return subtree
    index = path[0]
    children = list(node.children)
    children[index] = _replace_path(children[index], path[1:], subtree)
    return _rebuild(node, children)


def _rebuild(node: IRNode, children: Sequence[IRNode]) -> IRNode:
    if isinstance(node, FeatureNode):
        return node
    if isinstance(node, ConstantNode):
        return node
    if isinstance(node, NumOpNode):
        return NumOpNode(node.op, tuple(children))
    if isinstance(node, CompareNode):
        return CompareNode(node.op, tuple(children))
    if isinstance(node, BoolOpNode):
        return BoolOpNode(node.op, tuple(children))
    if isinstance(node, ActionNode):
        return node
    if isinstance(node, IfActionNode):
        return IfActionNode(children[0], children[1], children[2])
    if isinstance(node, WhenActionNode):
        return WhenActionNode(children[0], children[1])
    if isinstance(node, CombineActionNode):
        return CombineActionNode(children[0], children[1])
    raise ValueError(f"cannot rebuild node {type(node).__name__!r}")


def subtree_crossover(
    parent_a: "PolicyIRGenome", parent_b: "PolicyIRGenome", rng: Any,
) -> "PolicyIRGenome | None":
    """Exchange compatible typed subtrees; revalidated offspring (step 34)."""
    if parent_a.grammar.version != parent_b.grammar.version:
        return None
    nodes_a = [entry for entry in _collect_nodes(parent_a.root) if entry[1] is not parent_a.root]
    nodes_b = [entry for entry in _collect_nodes(parent_b.root)]
    if not nodes_a or not nodes_b:
        return None
    for _ in range(8):
        path_a, node_a = rng.choice(nodes_a)
        compatible = [
            (path_b, node_b) for path_b, node_b in nodes_b
            if node_b.ir_type == node_a.ir_type
        ]
        if not compatible:
            continue
        path_b, node_b = rng.choice(compatible)
        new_root = _replace_path(parent_a.root, path_a, node_b)
        offspring = PolicyIRGenome(
            root=new_root, grammar=parent_a.grammar, max_steps=parent_a.max_steps,
        )
        if offspring.validate() == []:
            return offspring
    return None


def subtree_mutation(genome: "PolicyIRGenome", rng: Any) -> "PolicyIRGenome | None":
    """Replace one subtree with a grammar-valid expression (step 34)."""
    nodes = [entry for entry in _collect_nodes(genome.root) if entry[1] is not genome.root]
    if not nodes:
        return None
    for _ in range(8):
        path, node = rng.choice(nodes)
        replacement = genome.grammar.random_tree(
            rng, ir_type=node.ir_type, depth_limit=3,
        )
        new_root = _replace_path(genome.root, path, replacement)
        offspring = PolicyIRGenome(
            root=new_root, grammar=genome.grammar, max_steps=genome.max_steps,
        )
        if offspring.validate() == []:
            return offspring
    return None


def static_constant_value(node: IRNode) -> float | None:
    """Return the constant value of a statically-foldable numeric node, else None."""
    if isinstance(node, ConstantNode):
        return node.value
    if isinstance(node, FeatureNode):
        return None
    if isinstance(node, NumOpNode):
        values = [static_constant_value(operand) for operand in node.operands]
        if any(value is None for value in values):
            return None
        if node.op == "add":
            return values[0] + values[1]
        if node.op == "sub":
            return values[0] - values[1]
        if node.op == "mul":
            return values[0] * values[1]
        if node.op == "div":
            return values[0] / values[1] if values[1] != 0 else None
        if node.op == "min":
            return min(values)
        if node.op == "max":
            return max(values)
        if node.op == "clamp01":
            return min(max(values[0], 0.0), 1.0)
    return None


class PolicyIRGenome:
    """Genome wrapping a typed policy-IR tree (step 33: declarative, inspectable)."""

    kind = "policy_ir"

    def __init__(
        self,
        root: IRNode,
        grammar: TypedGrammar | None = None,
        *,
        max_steps: int = 512,
    ) -> None:
        self.root = root
        self.grammar = grammar or DEFAULT_GRAMMAR
        self.max_steps = max_steps

    def validate(self) -> list[str]:
        return self.grammar.validate(self.root)

    def content_payload(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "grammar_version": self.grammar.version,
            "max_steps": self.max_steps,
            "root": self.root.as_dict(),
        }

    def capabilities(self) -> frozenset[str]:
        names: set[str] = set()
        for _path, node in _collect_nodes(self.root):
            if isinstance(node, ActionNode):
                names.add(node.call.name)
        return frozenset(names)

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind, "grammar_version": self.grammar.version,
            "max_steps": self.max_steps, "root": self.root.as_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], grammar: TypedGrammar | None = None) -> "PolicyIRGenome":
        if data.get("kind") != cls.kind:
            raise ValueError(f"genome kind mismatch: {data.get('kind')!r}")
        return cls(
            node_from_dict(data["root"]),
            grammar=grammar,
            max_steps=int(data.get("max_steps", 512)),
        )

    def evaluate_actions(
        self, context: Mapping[str, float],
    ) -> tuple[tuple[ActionCall, ...], int, bool]:
        """Interpret the policy IR under the bounded step budget (steps 12, 16).

        Returns ``(actions, steps, terminated)``; ``terminated`` marks a nonterminating
        or invalid evaluation that the sandbox records as INVALID.
        """
        budget = _StepBudget(self.max_steps)
        try:
            value = self.root.evaluate(context, budget)
        except _StepBudgetExceeded:
            return (), budget.used, True
        except (KeyError, ValueError, ArithmeticError):
            return (), budget.used, True
        if not isinstance(value, frozenset):
            return (), budget.used, True
        return tuple(sorted(value, key=lambda call: call.name)), budget.used, False

    def __eq__(self, other: object) -> bool:
        return isinstance(other, PolicyIRGenome) and other.content_payload() == self.content_payload()

    def __hash__(self) -> int:
        from ..domain.genome import content_digest
        return hash(content_digest(self.content_payload()))


__all__ = [
    "ACTION_OPS", "BOOL_OPS", "COMPARE_OPS", "DEFAULT_GRAMMAR", "GRAMMAR_VERSION",
    "IR_TYPES", "NUM_OPS", "POLICY_CONSTANTS", "POLICY_FEATURES", "ActionNode",
    "BoolOpNode", "CombineActionNode", "CompareNode", "ConstantNode", "FeatureNode",
    "IfActionNode", "IRNode", "NumOpNode", "PolicyIRGenome", "TypedGrammar",
    "WhenActionNode", "node_from_dict", "random_tree", "static_constant_value",
    "subtree_crossover", "subtree_mutation",
]
