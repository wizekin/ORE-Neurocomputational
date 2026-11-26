"""Goal and constraint formalization for the ORE Point 12 planning layer.

The adapter translates a natural-language goal into typed predicates and explicit
success conditions, preserves ambiguous interpretations as alternatives, and
extracts hard constraints and forbidden operations. Semantic interpretation is
delegated to an injectable hook so USG/HIC interfaces can be attached where
available without the planner duplicating ore_usg's semantic authority.
"""
from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from .contracts import GoalSpec, Predicate, SuccessCondition, planning_fingerprint

_CAPABILITY_KEYWORDS: tuple[tuple[str, str], ...] = (
    ("lexical", "language.lexical_access"),
    ("orthograph", "language.orthography"),
    ("syntactic", "language.syntax_parsing"),
    ("syntax", "language.syntax_parsing"),
    ("parse", "language.syntax_parsing"),
    ("semantic", "language.semantic_composition"),
    ("meaning", "language.semantic_composition"),
    ("discourse", "language.discourse_processing"),
    ("context", "language.discourse_processing"),
    ("comprehend", "language.comprehension"),
    ("understand", "language.comprehension"),
    ("visual", "vision.feature_extraction"),
    ("memory", "memory.integration"),
    ("consolidat", "memory.consolidation"),
    ("translate", "language.translation"),
    ("summar", "language.summarization"),
)

_CRITICAL_MARKERS = ("must", "critical", "required", "verify", "verified", "exactly")
_FORBIDDEN_PATTERNS = (
    re.compile(r"\bwithout\s+([a-z_][a-z0-9_.-]*)", re.IGNORECASE),
    re.compile(r"\bnever\s+([a-z_][a-z0-9_.-]*)", re.IGNORECASE),
    re.compile(r"\bforbid(?:den|s)?\s+([a-z_][a-z0-9_.-]*)", re.IGNORECASE),
)


@dataclass(frozen=True, slots=True)
class GoalFormalization:
    """Hold one formalization outcome with every preserved goal interpretation."""

    goal_id: str
    interpretations: tuple[GoalSpec, ...]
    ambiguous: bool
    clarification_request: str | None
    workspace_revision: int
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Require at least one interpretation and normalize ambiguity flags."""
        if not self.interpretations:
            raise ValueError("a formalization requires at least one goal interpretation")
        if len({item.goal_id for item in self.interpretations}) != len(self.interpretations):
            raise ValueError("goal interpretations must use distinct goal identifiers")
        if len({item.interpretation_label for item in self.interpretations}) != len(
            self.interpretations
        ):
            raise ValueError("goal interpretation labels must be unique")
        if self.workspace_revision < 0:
            raise ValueError("workspace_revision cannot be negative")
        object.__setattr__(self, "interpretations", tuple(self.interpretations))
        object.__setattr__(self, "notes", tuple(self.notes))

    @property
    def primary(self) -> GoalSpec:
        """Return the highest-priority interpretation under deterministic ordering."""
        return self.interpretations[0]

    def to_dict(self) -> dict[str, Any]:
        """Return a detached, JSON-compatible formalization record."""
        return {
            "goal_id": self.goal_id,
            "interpretations": [item.to_dict() for item in self.interpretations],
            "ambiguous": self.ambiguous,
            "clarification_request": self.clarification_request,
            "workspace_revision": self.workspace_revision,
            "notes": list(self.notes),
        }


def _match_capabilities(text: str) -> tuple[str, ...]:
    """Return every capability family mentioned by the goal text in match order."""
    matched: list[str] = []
    lowered = text.casefold()
    for keyword, capability in _CAPABILITY_KEYWORDS:
        if keyword in lowered and capability not in matched:
            matched.append(capability)
    return tuple(matched)


def _extract_forbidden(text: str) -> tuple[str, ...]:
    """Extract forbidden operators from explicit prohibition phrases."""
    forbidden: list[str] = []
    for pattern in _FORBIDDEN_PATTERNS:
        for match in pattern.finditer(text):
            token = match.group(1).casefold().replace("-", "_").replace(".", "_")
            if token not in forbidden:
                forbidden.append(token)
    return tuple(forbidden)


class GoalConstraintAdapter:
    """Translate natural-language goals into typed, versioned goal specifications."""

    def __init__(
        self,
        semantic_interpreter: Callable[[str], Sequence[Predicate]] | None = None,
    ) -> None:
        """Attach an optional USG/HIC-backed predicate interpreter hook."""
        self.semantic_interpreter = semantic_interpreter

    def _interpret_predicates(self, goal_text: str) -> tuple[Predicate, ...]:
        """Use the injected interpreter when available, else the documented rules."""
        if self.semantic_interpreter is not None:
            predicates = tuple(self.semantic_interpreter(goal_text))
            if predicates:
                return predicates
        capabilities = _match_capabilities(goal_text)
        return tuple(
            Predicate("goal.requires", (capability.replace(".", "_"),))
            for capability in capabilities
        ) or (Predicate("goal.requires", ("unspecified",)),)

    def formalize(
        self,
        goal_text: str,
        *,
        scope: str = "",
        deadline_seconds: float | None = None,
        quality_threshold: float = 0.75,
        workspace_revision: int = 0,
        goal_id: str | None = None,
        forbidden_operations: Sequence[str] = (),
    ) -> GoalFormalization:
        """Convert one goal into typed predicates, success conditions, and constraints."""
        text = goal_text.strip()
        if not text:
            raise ValueError("goal_text must be non-empty")
        resolved_goal_id = goal_id or f"goal_{planning_fingerprint(text)[:16]}"
        forbidden = tuple(dict.fromkeys((*_extract_forbidden(text), *forbidden_operations)))
        predicates = self._interpret_predicates(text)
        critical = any(marker in text.casefold() for marker in _CRITICAL_MARKERS)
        notes: list[str] = []

        interpretations: list[GoalSpec] = []
        capability_families = {
            "language.comprehension": (
                "language.lexical_access", "language.syntax_parsing",
                "language.semantic_composition",
            ),
        }
        mentioned = _match_capabilities(text)
        if mentioned:
            interpretations.append(self._build_goal(
                resolved_goal_id, text, predicates, mentioned,
                critical, deadline_seconds, forbidden,
            ))
            comprehension_family = capability_families["language.comprehension"]
            if mentioned == comprehension_family and "language.comprehension" not in mentioned:
                # The three comprehension components also form one aggregate capability.
                interpretations.append(self._build_goal(
                    f"{resolved_goal_id}_aggregate", text, predicates,
                    ("language.comprehension",), critical, deadline_seconds,
                    forbidden, label="aggregate_capability",
                ))
        else:
            notes.append("no capability keyword matched; goal kept as unspecific")
            interpretations.append(self._build_goal(
                resolved_goal_id, text, predicates, ("unspecified",),
                critical, deadline_seconds, forbidden,
            ))

        unique = {item.goal_id: item for item in interpretations}
        interpretations = [unique[key] for key in sorted(unique)]
        ambiguous = len(interpretations) > 1
        return GoalFormalization(
            goal_id=resolved_goal_id,
            interpretations=tuple(interpretations),
            ambiguous=ambiguous,
            clarification_request=(
                f"The goal maps to {len(interpretations)} interpretations; "
                "confirm which capability scope is intended."
                if ambiguous else None
            ),
            workspace_revision=workspace_revision,
            notes=tuple(notes),
        )

    @staticmethod
    def _build_goal(
        goal_id: str,
        description: str,
        predicates: tuple[Predicate, ...],
        capabilities: tuple[str, ...],
        critical: bool,
        deadline_seconds: float | None,
        forbidden: tuple[str, ...],
        *,
        label: str = "primary",
    ) -> GoalSpec:
        """Assemble one GoalSpec with one success condition per capability."""
        conditions = tuple(
            SuccessCondition(
                predicate=Predicate("goal.satisfied", (capability,)),
                weight=1.0,
                critical=critical,
            )
            for capability in capabilities
        )
        return GoalSpec(
            goal_id=goal_id,
            description=description,
            predicates=predicates,
            success_conditions=conditions,
            priority=0.5,
            deadline_seconds=deadline_seconds,
            critical_constraints=(
                ("all mentioned capabilities verified",) if critical else ()
            ),
            forbidden_operations=forbidden,
            interpretation_label=label,
        )


__all__ = ["GoalConstraintAdapter", "GoalFormalization"]
