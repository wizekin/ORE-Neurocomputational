"""Typed, provenance-preserving contracts for cross-structure memory exchange."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
import math
from typing import Any, Mapping, Sequence


def _plain(value: Any) -> Any:
    """Convert nested records and containers into detached JSON-compatible values."""
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _plain(value.to_dict())
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, set):
        return sorted((_plain(item) for item in value), key=repr)
    if hasattr(value, "item") and callable(value.item):
        return _plain(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _unit(name: str, value: float) -> float:
    """Validate a finite normalized score in the closed interval [0, 1]."""
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return number


@dataclass(frozen=True, slots=True)
class MemoryEvidence:
    """Normalize one long-term memory while retaining source, verification, and derivation links.

    relevance_score is a ranking score, not a probability of truth. confidence is retained
    separately and verification_status is never promoted by retrieval.
    """

    memory_id: str
    memory_type: str
    content: Any
    relevance_score: float
    confidence: float
    entities: Mapping[str, Any] = field(default_factory=dict)
    relations: Sequence[Any] = ()
    context: Mapping[str, Any] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)
    source_reference: str = "unspecified"
    verification_status: str = "unknown"
    valid_time: Any = None
    independence_group: str = ""
    parent_memory_ids: Sequence[str] = ()
    sequence_id: str = ""

    def __post_init__(self) -> None:
        """Validate identity and scores, then detach all caller-owned nested values."""
        for name in ("memory_id", "memory_type", "source_reference", "verification_status"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be non-empty text")
            object.__setattr__(self, name, value.strip())
        object.__setattr__(self, "memory_type", self.memory_type.casefold())
        if self.memory_type not in {"episodic", "semantic", "lexical"}:
            raise ValueError("memory_type must be episodic, semantic, or lexical")
        object.__setattr__(self, "relevance_score", _unit("relevance_score", self.relevance_score))
        object.__setattr__(self, "confidence", _unit("confidence", self.confidence))
        for name in ("entities", "context", "provenance"):
            value = getattr(self, name)
            if not isinstance(value, Mapping):
                raise TypeError(f"{name} must be a mapping")
            object.__setattr__(self, name, deepcopy(dict(value)))
        object.__setattr__(self, "content", _plain(self.content))
        object.__setattr__(self, "relations", tuple(_plain(item) for item in self.relations))
        object.__setattr__(self, "parent_memory_ids", tuple(str(item) for item in self.parent_memory_ids))
        object.__setattr__(self, "independence_group", self.independence_group.strip() or self.memory_id)
        object.__setattr__(self, "sequence_id", self.sequence_id.strip())

    def to_dict(self) -> dict[str, Any]:
        """Return a detached evidence record suitable for workspace publication."""
        return _plain(asdict(self))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "MemoryEvidence":
        """Rehydrate normalized evidence from a workspace record without forwarding envelope fields."""
        names = {
            "memory_id", "memory_type", "content", "relevance_score", "confidence",
            "entities", "relations", "context", "provenance", "source_reference",
            "verification_status", "valid_time", "independence_group",
            "parent_memory_ids", "sequence_id",
        }
        data = {key: value[key] for key in names if key in value}
        return cls(**data)


@dataclass(frozen=True, slots=True)
class MemoryQuery:
    """Represent multi-cue recall intent derived only from supplied or computed evidence."""

    sequence_id: str
    text: str
    goal: str = "understand_sentence"
    lexical_cues: Sequence[str] = ()
    event_cues: Mapping[str, Any] = field(default_factory=dict)
    context_cues: Mapping[str, Any] = field(default_factory=dict)
    temporal_cues: Mapping[str, Any] = field(default_factory=dict)
    current_timestamp: float | None = None
    retrieval_limit: int = 12

    def __post_init__(self) -> None:
        """Validate query identity and detach cue collections before ranking."""
        if not self.sequence_id.strip() or not self.text.strip() or not self.goal.strip():
            raise ValueError("sequence_id, text, and goal must be non-empty")
        if self.retrieval_limit < 1:
            raise ValueError("retrieval_limit must be positive")
        for name in ("event_cues", "context_cues", "temporal_cues"):
            value = getattr(self, name)
            if not isinstance(value, Mapping):
                raise TypeError(f"{name} must be a mapping")
            object.__setattr__(self, name, deepcopy(dict(value)))
        object.__setattr__(self, "lexical_cues", tuple(str(item) for item in self.lexical_cues if str(item).strip()))
        if self.current_timestamp is not None and not math.isfinite(float(self.current_timestamp)):
            raise ValueError("current_timestamp must be finite")


@dataclass(frozen=True, slots=True)
class MemoryIntegrationConfig:
    """Configure transparent retrieval weights, resource limits, and bounded feedback."""

    episodic_weights: Mapping[str, float] = field(default_factory=lambda: {
        "lexical": 0.30, "relations": 0.20, "context": 0.15,
        "goal": 0.15, "temporal": 0.05, "strength": 0.15,
    })
    semantic_weights: Mapping[str, float] = field(default_factory=lambda: {
        "concept": 0.35, "relations": 0.20, "goal": 0.15,
        "quality": 0.15, "context": 0.15,
    })
    max_retrieval_items: int = 12
    max_active_memories: int = 8
    max_integration_rounds: int = 2
    minimum_relevance: float = 0.01
    evidence_conflict_penalty: float = 0.25
    unresolved_penalty: float = 0.20
    memory_database_path: str = "ore_memory.sqlite3"

    def __post_init__(self) -> None:
        """Require non-negative, finite weights and explicit bounds on retrieval and recurrence."""
        for name in ("episodic_weights", "semantic_weights"):
            raw = getattr(self, name)
            if not isinstance(raw, Mapping) or not raw:
                raise ValueError(f"{name} must be a non-empty mapping")
            values = {str(key): float(value) for key, value in raw.items()}
            if any(not math.isfinite(item) or item < 0.0 for item in values.values()):
                raise ValueError(f"{name} values must be finite and non-negative")
            if sum(values.values()) <= 0.0:
                raise ValueError(f"{name} must have a positive total weight")
            object.__setattr__(self, name, values)
        if self.max_retrieval_items < 1 or self.max_active_memories < 1:
            raise ValueError("memory limits must be positive")
        if not 1 <= self.max_integration_rounds <= 2:
            raise ValueError("max_integration_rounds must be 1 or 2")
        object.__setattr__(self, "minimum_relevance", _unit("minimum_relevance", self.minimum_relevance))
        object.__setattr__(self, "evidence_conflict_penalty", _unit("evidence_conflict_penalty", self.evidence_conflict_penalty))
        object.__setattr__(self, "unresolved_penalty", _unit("unresolved_penalty", self.unresolved_penalty))
        if not isinstance(self.memory_database_path, str) or not self.memory_database_path.strip():
            raise ValueError("memory_database_path must be non-empty")


@dataclass(frozen=True, slots=True)
class MemoryBatch:
    """Collect ranked episodic and semantic evidence with transparent retrieval diagnostics."""

    query: MemoryQuery
    items: Sequence[MemoryEvidence]
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Freeze result order and detach diagnostics."""
        object.__setattr__(self, "items", tuple(self.items))
        object.__setattr__(self, "diagnostics", deepcopy(dict(self.diagnostics)))

    def to_dict(self) -> dict[str, Any]:
        """Serialize the query, ranked evidence, and score diagnostics."""
        return {
            "query": _plain(asdict(self.query)),
            "items": [item.to_dict() for item in self.items],
            "diagnostics": _plain(self.diagnostics),
        }


@dataclass(frozen=True, slots=True)
class MemoryHandoff:
    """Describe one module-specific delivery without converting hypotheses into observations."""

    module_id: str
    workspace_id: str
    workspace_revision: int
    sequence_id: str
    memory_ids: Sequence[str]
    payload: Mapping[str, Any]
    event_id: str = ""

    def __post_init__(self) -> None:
        """Validate routing identity and freeze the module payload."""
        for name in ("module_id", "workspace_id", "sequence_id"):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} must be non-empty")
        if self.workspace_revision < 0:
            raise ValueError("workspace_revision cannot be negative")
        object.__setattr__(self, "memory_ids", tuple(str(item) for item in self.memory_ids))
        object.__setattr__(self, "payload", deepcopy(dict(self.payload)))

    def to_dict(self) -> dict[str, Any]:
        """Return a detached handoff envelope for subscribers and caller inspection."""
        return _plain(asdict(self))


@dataclass(frozen=True, slots=True)
class MemoryIntegrationResult:
    """Expose both working-memory passes, retrieved references, routing, and encoding decisions."""

    workspace_id: str
    sequence_id: str
    initial_working_memory: Mapping[str, Any]
    integrated_working_memory: Mapping[str, Any]
    memory_batch: MemoryBatch
    workspace_snapshot: Mapping[str, Any]
    handoffs: Mapping[str, Any]
    encoding: Mapping[str, Any]
    diagnostics: Mapping[str, Any]

    def __post_init__(self) -> None:
        """Detach nested result data so later workspace changes cannot alter this result."""
        for name in (
            "initial_working_memory", "integrated_working_memory", "workspace_snapshot",
            "handoffs", "encoding", "diagnostics",
        ):
            object.__setattr__(self, name, deepcopy(dict(getattr(self, name))))

    def to_dict(self) -> dict[str, Any]:
        """Return the integrated result as JSON-compatible data."""
        return _plain(asdict(self))
