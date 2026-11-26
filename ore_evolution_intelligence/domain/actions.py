"""Approved ORE action vocabulary and simulated coordinator modules (steps 8, 11, 31).

Genotypes are restricted to an *explicit action vocabulary* (step 8): every action a
candidate strategy can express must be declared here with a typed parameter schema.
Unknown or privileged actions (filesystem, network, shell, database mutation,
unrestricted imports) are structurally absent from the vocabulary and rejected by the
sandbox whenever they appear (acceptance tests 2, 8, 16).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Mapping

#: Simulated ORE coordinator modules that candidate strategies may activate.
#: ``verification_monitor`` models the CROS safety/verification monitor the ORE
#: integration map places next to AdaptiveResourceCoordinator.
ORE_MODULES: tuple[str, ...] = (
    "ore_lexical_access",
    "ore_syntax",
    "ore_semantic",
    "ore_semantic_composition",
    "ore_discourse",
    "ore_working_memory",
    "ore_vwfa",
    "ore_reading",
    "ore_hippocampal_encoding",
    "ore_sleep_consolidation",
    "verification_monitor",
)

#: Task routing queues exposed by the simulated ModuleScheduler.
ROUTE_QUEUES: tuple[str, ...] = ("fast", "standard", "deep")

#: Deferral reasons accepted by the scheduler action vocabulary.
DEFER_REASONS: tuple[str, ...] = ("low_confidence", "resource_pressure", "safety_review")

#: Escalation modes for the verification monitor (discrete parameter).
ESCALATION_MODES: tuple[str, ...] = ("conservative", "balanced", "aggressive")


@dataclass(frozen=True, slots=True)
class ParameterSpec:
    """Typed parameter schema entry for one action (step 8: typed parameters)."""

    name: str
    kind: str  # "module" | "queue" | "defer_reason" | "escalation_mode" | "int" | "number" | "str"
    minimum: float | None = None
    maximum: float | None = None
    choices: tuple[str, ...] = field(default_factory=tuple)

    def validate(self, value: Any) -> str | None:
        """Return an error message for invalid values, else ``None``."""
        if self.kind == "module":
            ok = isinstance(value, str) and value in ORE_MODULES
        elif self.kind == "queue":
            ok = isinstance(value, str) and value in ROUTE_QUEUES
        elif self.kind == "defer_reason":
            ok = isinstance(value, str) and value in DEFER_REASONS
        elif self.kind == "escalation_mode":
            ok = isinstance(value, str) and value in ESCALATION_MODES
        elif self.kind == "int":
            if not isinstance(value, int) or isinstance(value, bool):
                return f"parameter {self.name!r} must be an int"
            ok = (
                self.minimum is None or value >= self.minimum
            ) and (self.maximum is None or value <= self.maximum)
        elif self.kind == "number":
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return f"parameter {self.name!r} must be numeric"
            if isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
                return f"parameter {self.name!r} must be finite"
            ok = (
                self.minimum is None or value >= self.minimum
            ) and (self.maximum is None or value <= self.maximum)
        else:  # free string bounded by choices
            ok = (not self.choices) or (isinstance(value, str) and value in self.choices)
        if not ok:
            return (
                f"parameter {self.name!r} value {value!r} violates schema "
                f"(kind={self.kind}, min={self.minimum}, max={self.maximum})"
            )
        return None

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "kind": self.kind,
            "minimum": self.minimum, "maximum": self.maximum,
            "choices": list(self.choices),
        }


#: Approved action vocabulary: action name -> typed parameter schema (step 8).
#: Anything outside this mapping is prohibited for candidate strategies.
ACTION_VOCABULARY: dict[str, tuple[ParameterSpec, ...]] = {
    "activate_module": (ParameterSpec("module", "module", choices=ORE_MODULES),),
    "deactivate_module": (ParameterSpec("module", "module", choices=ORE_MODULES),),
    "route_task": (ParameterSpec("queue", "queue", choices=ROUTE_QUEUES),),
    "defer_task": (ParameterSpec("reason", "defer_reason", choices=DEFER_REASONS),),
    "escalate_verification": (),
    "request_retrieval": (
        ParameterSpec("depth", "int", minimum=0, maximum=3),
    ),
    "adjust_threshold": (
        ParameterSpec("parameter", "str", choices=("activation", "verification")),
        ParameterSpec("value", "number", minimum=0.0, maximum=1.0),
    ),
    "report_abstention": (ParameterSpec("reason", "str", choices=("low_confidence", "incomplete_evidence")),),
}

#: Explicitly prohibited action families (step 8; section 9 safeguards). The sandbox
#: rejects these names (and anything not in ACTION_VOCABULARY) outright.
PROHIBITED_ACTION_PREFIXES: tuple[str, ...] = (
    "fs_", "net_", "shell_", "db_", "import_", "exec_", "eval_",
)

PROHIBITED_ACTIONS: tuple[str, ...] = (
    "fs_write_file", "fs_read_file", "net_http_get", "net_connect",
    "shell_execute", "db_mutate_usg", "import_arbitrary", "exec_python_source",
)


@dataclass(frozen=True, slots=True)
class ActionCall:
    """One approved ORE operation with validated typed parameters."""

    name: str
    params: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        errors = validate_action(self.name, self.params)
        if errors:
            raise ValueError(f"invalid action call: {errors[0]}")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ActionCall):
            return NotImplemented
        return self.name == other.name and dict(self.params) == dict(other.params)

    def __hash__(self) -> int:
        """Hash on the canonical parameter encoding (params is a dict)."""
        return hash((
            self.name,
            json.dumps(dict(self.params), sort_keys=True, separators=(",", ":"), default=str),
        ))

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "params": dict(self.params)}

    def content_payload(self) -> dict[str, Any]:
        return {"name": self.name, "params": dict(sorted(self.params.items()))}


def validate_action(name: str, params: Mapping[str, Any]) -> list[str]:
    """Validate one action against the vocabulary; return error messages."""
    errors: list[str] = []
    if not isinstance(name, str) or not name:
        return ["action name must be a non-empty string"]
    if any(name.startswith(prefix) for prefix in PROHIBITED_ACTION_PREFIXES):
        errors.append(f"action {name!r} belongs to a prohibited family")
    if name not in ACTION_VOCABULARY:
        errors.append(f"action {name!r} is not in the approved vocabulary")
        return errors
    specs = ACTION_VOCABULARY[name]
    known = {spec.name for spec in specs}
    for key in params:
        if key not in known:
            errors.append(f"action {name!r} has unknown parameter {key!r}")
    for spec in specs:
        if spec.name not in params:
            errors.append(f"action {name!r} is missing required parameter {spec.name!r}")
            continue
        problem = spec.validate(params[spec.name])
        if problem:
            errors.append(problem)
    return errors


def is_approved_action(name: str) -> bool:
    """Return True only for names present in the approved vocabulary."""
    return name in ACTION_VOCABULARY


__all__ = [
    "ACTION_VOCABULARY", "DEFER_REASONS", "ESCALATION_MODES", "ORE_MODULES",
    "PROHIBITED_ACTIONS", "PROHIBITED_ACTION_PREFIXES", "ROUTE_QUEUES",
    "ActionCall", "ParameterSpec", "is_approved_action", "validate_action",
]
