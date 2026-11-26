"""Phase IV, Step 16 - convert human intent into a typed task request.

Preserve the exact original request, extract candidate goals, entities,
constraints and expected outputs, resolve known terms through USG, mark
ambiguity and missing information, and build the typed task tuple
``T = (G, C, I, O, B, P)`` - goals, constraints, inputs, outputs, resource
budget and permission policy.  A clarification is requested only when a
missing fact materially changes the execution plan; otherwise the assumption
is explicitly bounded and recorded.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from ore_contracts import ComputeBudget

from .core import new_id, sha256_digest, utc_now_iso

TASK_REQUEST_SCHEMA = "ore.point24.task_request.v1"
TASK_REQUEST_CONTRACT = "P24-TASK-REQUEST"


@dataclass(frozen=True, slots=True)
class ClarificationRequest:
    """A clarification requirement raised by intent interpretation."""

    question: str
    reason: str
    materially_changes_plan: bool = True


@dataclass(frozen=True, slots=True)
class TaskRequest:
    """The typed Step 16 request T = (G, C, I, O, B, P)."""

    goal: str
    constraints: tuple[str, ...]
    inputs: tuple[str, ...]
    expected_outputs: tuple[str, ...]
    resource_budget: ComputeBudget
    permission_policy: str
    original_request: str
    task_request_id: str = field(default_factory=lambda: new_id("point24_request"))
    required_capabilities: tuple[str, ...] = ()
    optional_capabilities: tuple[str, ...] = ()
    known_term_resolutions: Mapping[str, str] = field(default_factory=dict)
    ambiguities: tuple[str, ...] = ()
    unsupported_assumptions: tuple[str, ...] = ()
    bounded_assumptions: tuple[str, ...] = ()
    clarifications: tuple[ClarificationRequest, ...] = ()
    created_at: str = field(default_factory=utc_now_iso)

    def __post_init__(self) -> None:
        if not self.goal.strip():
            raise ValueError("goal must be non-empty")
        if not self.original_request.strip():
            raise ValueError("original_request must preserve the exact user input")

    @property
    def needs_clarification(self) -> bool:
        return any(item.materially_changes_plan for item in self.clarifications)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": TASK_REQUEST_SCHEMA,
            "task_request_id": self.task_request_id,
            "goal": self.goal,
            "constraints": list(self.constraints),
            "inputs": list(self.inputs),
            "expected_outputs": list(self.expected_outputs),
            "resource_budget": {
                "wall_clock_seconds": self.resource_budget.wall_clock_seconds,
                "max_memory_mb": self.resource_budget.max_memory_mb,
                "max_recurrent_iterations": self.resource_budget.max_recurrent_iterations,
            },
            "permission_policy": self.permission_policy,
            "original_request": self.original_request,
            "required_capabilities": list(self.required_capabilities),
            "optional_capabilities": list(self.optional_capabilities),
            "known_term_resolutions": dict(self.known_term_resolutions),
            "ambiguities": list(self.ambiguities),
            "unsupported_assumptions": list(self.unsupported_assumptions),
            "bounded_assumptions": list(self.bounded_assumptions),
            "clarifications": [
                {"question": item.question, "reason": item.reason,
                 "materially_changes_plan": item.materially_changes_plan}
                for item in self.clarifications
            ],
            "needs_clarification": self.needs_clarification,
            "created_at": self.created_at,
        }

    def digest(self) -> str:
        return sha256_digest(self.to_dict())


class IntentInterpreter:
    """A deterministic first-version intent interpreter for Step 16.

    The interpreter never invents requirements: unknown capability terms are
    marked as unsupported assumptions, ambiguous phrasing is preserved as an
    ambiguity marker, and free-form text is never treated as a substitute for
    a declared module capability.
    """

    GOAL_VERBS = ("analyze", "plan", "verify", "integrate", "summarize", "search", "evaluate", "explain")

    def __init__(self, known_terms: Mapping[str, str] | None = None) -> None:
        self._known_terms = dict(known_terms or {})

    def interpret(
        self,
        request: str,
        *,
        required_capabilities: tuple[str, ...] = (),
        optional_capabilities: tuple[str, ...] = (),
        constraints: tuple[str, ...] = (),
        expected_outputs: tuple[str, ...] = (),
        budget: ComputeBudget | None = None,
        permission_policy: str = "least_privilege",
        known_terms: Mapping[str, str] | None = None,
    ) -> TaskRequest:
        """Build the typed task request from the original human input."""
        text = str(request).strip()
        if not text:
            raise ValueError("request must be non-empty")
        terms = dict(self._known_terms)
        terms.update(known_terms or {})

        resolved: dict[str, str] = {}
        unsupported: list[str] = []
        lowered = text.casefold()
        for term, resolution in terms.items():
            if term.casefold() in lowered:
                resolved[term] = resolution
        for capability in required_capabilities:
            if "." not in capability:
                unsupported.append(
                    f"capability {capability!r} is not a declared namespaced capability"
                )

        ambiguities: list[str] = []
        if any(word in lowered for word in ("maybe", "perhaps", "or something", "somehow")):
            ambiguities.append("request contains hedged phrasing; interpretation is uncertain")

        clarifications: list[ClarificationRequest] = []
        if not expected_outputs:
            clarifications.append(ClarificationRequest(
                question="What output artifact does the request expect?",
                reason="expected outputs are empty; the execution plan depends on them",
                materially_changes_plan=True,
            ))

        goal = text
        for verb in self.GOAL_VERBS:
            marker = f"{verb} "
            if lowered.startswith(verb) and marker in text.casefold() + " ":
                goal = text
                break

        return TaskRequest(
            goal=goal,
            constraints=tuple(constraints),
            inputs=(text,),
            expected_outputs=tuple(expected_outputs),
            resource_budget=budget or ComputeBudget(),
            permission_policy=permission_policy,
            original_request=text,
            required_capabilities=tuple(required_capabilities),
            optional_capabilities=tuple(optional_capabilities),
            known_term_resolutions=resolved,
            ambiguities=tuple(ambiguities),
            unsupported_assumptions=tuple(unsupported),
            bounded_assumptions=(
                "no expected outputs declared; proceeding with a goal-conditioned plan",
            ) if not expected_outputs and not any(c.materially_changes_plan for c in clarifications) else (),
            clarifications=tuple(clarifications),
        )


__all__ = [
    "TASK_REQUEST_CONTRACT",
    "TASK_REQUEST_SCHEMA",
    "ClarificationRequest",
    "IntentInterpreter",
    "TaskRequest",
]
