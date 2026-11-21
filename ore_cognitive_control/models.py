"""Typed contracts for the Point 02 cognitive-control foundation."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Mapping, Protocol, Sequence


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class StepStatus(str, Enum):
    SELECTED = "selected"
    EXECUTED = "executed"
    NO_OPERATOR = "no_operator"
    BUDGET_EXHAUSTED = "budget_exhausted"
    PRECONDITION_FAILED = "precondition_failed"
    VALIDATION_FAILED = "validation_failed"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class DecisionBudget:
    max_operations: int = 1
    max_elapsed_ms: float = 250.0
    max_retrieval_candidates: int = 20

    def __post_init__(self) -> None:
        if self.max_operations < 0:
            raise ValueError("max_operations must be >= 0")
        if self.max_elapsed_ms < 0:
            raise ValueError("max_elapsed_ms must be >= 0")
        if self.max_retrieval_candidates < 0:
            raise ValueError("max_retrieval_candidates must be >= 0")


@dataclass(frozen=True, slots=True)
class CognitiveContext:
    """Detached state snapshot passed into one bounded decision cycle."""
    context_id: str
    goal: str
    state: Mapping[str, Any] = field(default_factory=dict)
    cues: tuple[str, ...] = ()
    task_relevance: Mapping[str, float] = field(default_factory=dict)
    available_capabilities: frozenset[str] = frozenset()
    revision: int = 0

    def __post_init__(self) -> None:
        if not self.context_id.strip():
            raise ValueError("context_id must be non-empty")
        if not self.goal.strip():
            raise ValueError("goal must be non-empty")
        if self.revision < 0:
            raise ValueError("revision must be >= 0")
        object.__setattr__(self, "state", dict(self.state))
        object.__setattr__(self, "cues", tuple(self.cues))
        object.__setattr__(self, "task_relevance", dict(self.task_relevance))
        object.__setattr__(self, "available_capabilities", frozenset(self.available_capabilities))


@dataclass(frozen=True, slots=True)
class RetrievalQuery:
    cues: tuple[str, ...] = ()
    limit: int = 20
    now: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        if self.limit < 0:
            raise ValueError("limit must be >= 0")


@dataclass(frozen=True, slots=True)
class RetrievalCandidate:
    memory_id: str
    content: Any
    access_timestamps: tuple[datetime, ...] = ()
    cue_terms: frozenset[str] = frozenset()
    task_tags: frozenset[str] = frozenset()
    provenance_score: float | None = None
    estimated_cost: float = 0.0
    available: bool = True

    def __post_init__(self) -> None:
        if not self.memory_id.strip():
            raise ValueError("memory_id must be non-empty")
        if self.estimated_cost < 0:
            raise ValueError("estimated_cost must be >= 0")
        if self.provenance_score is not None and not 0 <= self.provenance_score <= 1:
            raise ValueError("provenance_score must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    candidate: RetrievalCandidate
    activation: float
    cue_match: float
    task_relevance: float
    provenance: float | None
    total_score: float


@dataclass(frozen=True, slots=True)
class OperatorResult:
    success: bool
    progress: float = 0.0
    state_patch: Mapping[str, Any] = field(default_factory=dict)
    message: str = ""
    evidence: tuple[Mapping[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if not 0 <= self.progress <= 1:
            raise ValueError("progress must be in [0, 1]")
        object.__setattr__(self, "state_patch", dict(self.state_patch))
        object.__setattr__(self, "evidence", tuple(dict(x) for x in self.evidence))


class Operator(Protocol):
    operator_id: str
    version: str
    required_capabilities: frozenset[str]
    estimated_cost: float

    def precondition(self, context: CognitiveContext) -> bool: ...
    def execute(self, context: CognitiveContext) -> OperatorResult: ...


@dataclass(frozen=True, slots=True)
class CognitiveTrace:
    context_id: str
    status: StepStatus
    selected_operator: str | None
    candidates_considered: int
    operations_used: int
    elapsed_ms: float
    reason: str
    result: OperatorResult | None = None
    timestamp: datetime = field(default_factory=utc_now)


TraceSink = Callable[[CognitiveTrace], None]
