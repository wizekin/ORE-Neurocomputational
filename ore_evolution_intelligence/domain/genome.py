"""Strategy genotypes: bounded parameter vectors and finite policy tables (step 7).

Point 9 starts with *bounded numeric parameter vectors and finite policy tables*
(step 7). Typed expression trees arrive through
:mod:`ore_evolution_intelligence.search.genetic_programming`, whose
``PolicyIRGenome`` satisfies the same genome protocol.

Genome protocol (duck-typed, used by every layer):

- ``kind``: stable genome kind tag (``"parameters"`` / ``"policy_table"`` / ``"policy_ir"``)
- ``validate()``: list of error strings (empty means valid)
- ``content_payload()``: canonical payload for content hashing
- ``as_dict()`` / ``from_dict()``: schema-valid serialisation (step 2.1)
- ``capabilities()``: frozenset of approved action names the genome can express
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any, Mapping

from .actions import (
    ACTION_VOCABULARY, DEFER_REASONS, ESCALATION_MODES, ORE_MODULES, ROUTE_QUEUES,
    ActionCall, validate_action,
)


def canonical_json(payload: Mapping[str, Any]) -> str:
    """Deterministic JSON encoding used for all content hashes and manifests."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def content_digest(payload: Mapping[str, Any]) -> str:
    """SHA-256 over the canonical JSON encoding of ``payload``."""
    return sha256(canonical_json(payload).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class ParameterDefinition:
    """One bounded search-space dimension (step 19: validated ranges)."""

    name: str
    kind: str  # "continuous" | "integer" | "choice"
    low: float = 0.0
    high: float = 1.0
    choices: tuple[str, ...] = field(default_factory=tuple)
    default: Any = None
    description: str = ""

    def __post_init__(self) -> None:
        if self.kind in ("continuous", "integer"):
            if not (self.low < self.high):
                raise ValueError(f"parameter {self.name!r} needs low < high")
            if self.kind == "integer" and (self.low != math.floor(self.low) or self.high != math.floor(self.high)):
                raise ValueError(f"integer parameter {self.name!r} needs integer bounds")
        elif self.kind == "choice":
            if not self.choices:
                raise ValueError(f"choice parameter {self.name!r} needs choices")
        else:
            raise ValueError(f"unknown parameter kind {self.kind!r}")

    def contains(self, value: Any) -> bool:
        if self.kind == "continuous":
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return False
            return math.isfinite(float(value)) and self.low <= value <= self.high
        if self.kind == "integer":
            return isinstance(value, int) and not isinstance(value, bool) and self.low <= value <= self.high
        return value in self.choices

    def validate(self, value: Any) -> str | None:
        if self.contains(value):
            return None
        return f"parameter {self.name!r} value {value!r} outside validated range"

    def clip(self, value: Any) -> Any:
        """Clip a numeric value into the validated range (safe perturbations)."""
        if self.kind == "continuous":
            return min(max(float(value), self.low), self.high)
        if self.kind == "integer":
            return int(min(max(int(round(value)), int(self.low)), int(self.high)))
        # choice: snap to nearest by index
        if value in self.choices:
            return value
        try:
            return self.choices[min(max(int(value), 0), len(self.choices) - 1)]
        except (TypeError, ValueError):
            return self.choices[0]

    def default_value(self) -> Any:
        if self.default is not None:
            return self.default
        if self.kind == "continuous":
            return (self.low + self.high) / 2.0
        if self.kind == "integer":
            return int(round((self.low + self.high) / 2.0))
        return self.choices[len(self.choices) // 2]

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "kind": self.kind, "low": self.low, "high": self.high,
            "choices": list(self.choices), "default": self.default,
            "description": self.description,
        }


def default_search_space() -> tuple[ParameterDefinition, ...]:
    """Bounded coordinator search space (steps 1, 19).

    Covers module-selection weights, activation thresholds, routing priorities and
    resource budgets for the simulated AdaptiveResourceCoordinator, plus one integer
    and one choice parameter so discrete mutation is exercised and CMA-ES's
    continuous-only guard (step 30) is testable.
    """
    definitions: list[ParameterDefinition] = []
    for module in ORE_MODULES:
        definitions.append(ParameterDefinition(
            f"weight__{module}", "continuous", low=0.0, high=2.0, default=0.6,
            description=f"module-selection weight for {module}",
        ))
        definitions.append(ParameterDefinition(
            f"threshold__{module}", "continuous", low=0.05, high=1.0, default=0.5,
            description=f"activation threshold for {module}",
        ))
    for queue in ROUTE_QUEUES:
        definitions.append(ParameterDefinition(
            f"route_weight__{queue}", "continuous", low=0.0, high=1.0, default=1.0 / 3.0,
            description=f"routing priority weight for the {queue} queue",
        ))
    definitions.append(ParameterDefinition(
        "cpu_budget_share", "continuous", low=0.1, high=1.0, default=0.6,
        description="share of coordinator CPU budget granted to candidate strategies",
    ))
    definitions.append(ParameterDefinition(
        "memory_budget_share", "continuous", low=0.1, high=1.0, default=0.6,
        description="share of coordinator memory budget granted to candidate strategies",
    ))
    definitions.append(ParameterDefinition(
        "retrieval_depth", "integer", low=0, high=3, default=1,
        description="bounded retrieval depth requested for ambiguous tasks",
    ))
    definitions.append(ParameterDefinition(
        "escalation_mode", "choice", choices=ESCALATION_MODES, default="balanced",
        description="verification-monitor escalation posture",
    ))
    return tuple(definitions)


class ParameterGenome:
    """Bounded numeric parameter vector genome (step 7, step 19).

    Values are always within the validated ranges after construction; ``validate``
    re-checks every value so deserialised or perturbed genomes must pass before use.
    """

    kind = "parameters"

    def __init__(
        self,
        values: Mapping[str, Any],
        space: tuple[ParameterDefinition, ...] | None = None,
    ) -> None:
        self.space: tuple[ParameterDefinition, ...] = space or default_search_space()
        self._by_name = {definition.name: definition for definition in self.space}
        unknown = set(values) - set(self._by_name)
        if unknown:
            raise ValueError(f"unknown parameters: {sorted(unknown)}")
        self.values: dict[str, Any] = {name: values[name] for name in self._by_name if name in values}
        missing = [name for name in self._by_name if name not in self.values]
        if missing:
            raise ValueError(f"missing parameters: {sorted(missing)}")

    # -- genome protocol ---------------------------------------------------
    def validate(self) -> list[str]:
        errors: list[str] = []
        for name, definition in self._by_name.items():
            problem = definition.validate(self.values.get(name))
            if problem:
                errors.append(problem)
        for name, value in self.values.items():
            if name not in self._by_name:
                errors.append(f"unknown parameter {name!r}")
        return errors

    def content_payload(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "space_version": content_digest({"defs": [d.as_dict() for d in self.space]}),
            "values": {name: self.values[name] for name in sorted(self.values)},
        }

    def capabilities(self) -> frozenset[str]:
        """Parameter genomes act through module activation and routing only."""
        return frozenset({"activate_module", "route_task", "request_retrieval"})

    def as_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "values": {name: self.values[name] for name in sorted(self.values)}}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ParameterGenome":
        if data.get("kind") != cls.kind:
            raise ValueError(f"genome kind mismatch: {data.get('kind')!r}")
        return cls(data["values"])

    # -- helpers ------------------------------------------------------------
    def get(self, name: str) -> Any:
        return self.values[name]

    def clone_with(self, updates: Mapping[str, Any]) -> "ParameterGenome":
        """Clone with validated/clipped updates (crossover and mutation helper)."""
        merged = dict(self.values)
        for name, value in updates.items():
            if name not in self._by_name:
                raise ValueError(f"unknown parameter {name!r}")
            merged[name] = self._by_name[name].clip(value)
        return ParameterGenome(merged, self.space)

    def baseline_genome(self) -> "ParameterGenome":
        """The default (baseline) genome for this search space."""
        return ParameterGenome(
            {definition.name: definition.default_value() for definition in self.space},
            self.space,
        )

    def normalized_vector(self) -> list[float]:
        """Values mapped to [0,1] per dimension (distances, CMA-ES seeding)."""
        normalized: list[float] = []
        for definition in self.space:
            value = self.values[definition.name]
            if definition.kind == "continuous":
                normalized.append((float(value) - definition.low) / (definition.high - definition.low))
            elif definition.kind == "integer":
                span = max(definition.high - definition.low, 1)
                normalized.append((int(value) - definition.low) / float(span))
            else:
                normalized.append(definition.choices.index(value) / max(len(definition.choices) - 1, 1))
        return normalized

    def continuous_names(self) -> list[str]:
        """Names of continuous parameters only (step 30: CMA-ES applicability)."""
        return [d.name for d in self.space if d.kind == "continuous"]

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ParameterGenome) and other.content_payload() == self.content_payload()

    def __hash__(self) -> int:
        return hash(content_digest(self.content_payload()))

    def __repr__(self) -> str:
        digest = content_digest(self.content_payload())
        return f"ParameterGenome({digest[:12]}, params={len(self.values)})"


#: Context schema for finite policy tables (step 7: finite policy tables).
POLICY_TABLE_CONTEXT_DIMENSIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("task_family", (
        "lexical_lookup", "semantic_composition", "syntax_parse",
        "reading", "discourse", "consolidation",
    )),
    ("difficulty", ("low", "medium", "high")),
    ("safety_critical", ("no", "yes")),
    ("resource_state", ("normal", "pressured")),
)

#: Bounded number of active modules per policy-table cell.
MAX_TABLE_CELL_ACTIONS = 6


def table_context_key(
    family: str, difficulty: str, safety_critical: bool, resource_state: str,
) -> tuple[str, ...]:
    """Build a canonical policy-table context key from task features."""
    return (family, difficulty, "yes" if safety_critical else "no", resource_state)


class PolicyTableGenome:
    """Finite policy-table genome: context key -> frozenset of approved actions.

    Table size is bounded by the context schema (6 families x 3 difficulties x
    2 safety flags x 2 resource states) and per-cell action count.
    """

    kind = "policy_table"
    max_cell_actions = MAX_TABLE_CELL_ACTIONS

    def __init__(
        self,
        rules: Mapping[tuple[str, ...], frozenset[ActionCall] | set[ActionCall]],
        context_dimensions: tuple[tuple[str, tuple[str, ...]], ...] = POLICY_TABLE_CONTEXT_DIMENSIONS,
    ) -> None:
        self.context_dimensions = context_dimensions
        self._valid_keys: set[tuple[str, ...]] = set()
        for family in context_dimensions[0][1]:
            for difficulty in context_dimensions[1][1]:
                for safety in context_dimensions[2][1]:
                    for resource in context_dimensions[3][1]:
                        self._valid_keys.add((family, difficulty, safety, resource))
        self.rules: dict[tuple[str, ...], frozenset[ActionCall]] = {
            tuple(key): frozenset(actions) for key, actions in rules.items()
        }

    def validate(self) -> list[str]:
        errors: list[str] = []
        for key in self.rules:
            if key not in self._valid_keys:
                errors.append(f"policy table key {key!r} does not match the context schema")
        for key, actions in self.rules.items():
            if len(actions) > self.max_cell_actions:
                errors.append(
                    f"policy cell {key!r} exceeds the action bound ({len(actions)} > {self.max_cell_actions})"
                )
            for action in actions:
                problems = validate_action(action.name, action.params)
                if problems:
                    errors.append(f"policy cell {key!r}: {problems[0]}")
        return errors

    def content_payload(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "context_version": content_digest(
                {"dims": [[name, list(values)] for name, values in self.context_dimensions]}
            ),
            "rules": {
                "|".join(key): sorted(a.content_payload()["name"] + "@" + canonical_json(a.content_payload()["params"]) for a in actions)
                for key, actions in sorted(self.rules.items())
            },
        }

    def capabilities(self) -> frozenset[str]:
        names: set[str] = set()
        for actions in self.rules.values():
            for action in actions:
                names.add(action.name)
        return frozenset(names)

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "rules": {
                "|".join(key): [action.as_dict() for action in sorted(actions, key=lambda a: a.name)]
                for key, actions in sorted(self.rules.items())
            },
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PolicyTableGenome":
        if data.get("kind") != cls.kind:
            raise ValueError(f"genome kind mismatch: {data.get('kind')!r}")
        rules: dict[tuple[str, ...], set[ActionCall]] = {}
        for key_text, action_dicts in data["rules"].items():
            key = tuple(key_text.split("|"))
            rules[key] = {ActionCall(a["name"], a.get("params", {})) for a in action_dicts}
        return cls(rules)

    def decision_for(self, key: tuple[str, ...]) -> frozenset[ActionCall]:
        """Actions chosen for one context key (default: defer under resource pressure)."""
        if key in self.rules:
            return self.rules[key]
        return frozenset({ActionCall("defer_task", {"reason": "low_confidence"})})

    def __eq__(self, other: object) -> bool:
        return isinstance(other, PolicyTableGenome) and other.content_payload() == self.content_payload()

    def __hash__(self) -> int:
        return hash(content_digest(self.content_payload()))


def actions_from_payload(payload: Mapping[str, Any]) -> frozenset[ActionCall]:
    """Rebuild a frozenset of validated action calls from a serialized payload."""
    calls: list[ActionCall] = []
    for entry in payload.get("actions", []):
        call = ActionCall(entry["name"], entry.get("params", {}))
        calls.append(call)
    return frozenset(calls)


__all__ = [
    "MAX_TABLE_CELL_ACTIONS", "POLICY_TABLE_CONTEXT_DIMENSIONS", "ActionCall",
    "ParameterDefinition", "ParameterGenome", "PolicyTableGenome",
    "actions_from_payload", "canonical_json", "content_digest",
    "default_search_space", "table_context_key",
]
