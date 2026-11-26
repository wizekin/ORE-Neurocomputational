"""Strategy versioning and immutable rollback targets (steps 3, 27; section 9).

The baseline (step 3) is captured as an *immutable rollback target*: current
deterministic policy, configuration, benchmark results and source version, preserved
so the coordinator can always restore the pre-experiment configuration (acceptance
tests 4, 34: baseline is immutable and replayable; rollback restores prior
configuration and records the event).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping

from ..domain.genome import content_digest

#: Change kinds recognized by the versioning log.
CHANGE_KINDS: tuple[str, ...] = ("initial", "improvement", "rollback", "deprecation")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


@dataclass(frozen=True, slots=True)
class RollbackTarget:
    """Immutable snapshot of the production configuration (step 3)."""

    strategy_id: str | None  # None means the deterministic baseline configuration
    content_hash: str
    configuration: Mapping[str, Any]
    benchmark_version: str
    benchmark_results: Mapping[str, Any]
    source_version: str
    captured_at_utc: str = field(default_factory=_utc_now)

    def as_dict(self) -> dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "content_hash": self.content_hash,
            "configuration": dict(self.configuration),
            "benchmark_version": self.benchmark_version,
            "benchmark_results": dict(self.benchmark_results),
            "source_version": self.source_version,
            "captured_at_utc": self.captured_at_utc,
        }

    def digest(self) -> str:
        return content_digest(self.as_dict())


@dataclass(frozen=True, slots=True)
class StrategyVersionInfo:
    """One entry in a strategy's version history."""

    strategy_id: str
    version: int
    content_hash: str
    change_kind: str
    registered_at_utc: str
    notes: str = ""

    def __post_init__(self) -> None:
        if self.change_kind not in CHANGE_KINDS:
            raise ValueError(f"unknown change kind {self.change_kind!r}")
        if self.version < 1:
            raise ValueError("version must be >= 1")

    def as_dict(self) -> dict[str, Any]:
        return {
            "strategy_id": self.strategy_id, "version": self.version,
            "content_hash": self.content_hash, "change_kind": self.change_kind,
            "registered_at_utc": self.registered_at_utc, "notes": self.notes,
        }


class Versioning:
    """Version history per strategy plus the pinned rollback target (steps 3, 27)."""

    def __init__(self) -> None:
        self._history: dict[str, list[StrategyVersionInfo]] = {}
        self._current: dict[str, StrategyVersionInfo] = {}
        self._rollback_target: RollbackTarget | None = None

    def pin_rollback_target(self, target: RollbackTarget) -> None:
        """Pin the immutable baseline rollback target (step 3; set once)."""
        if self._rollback_target is not None:
            raise ValueError("rollback target is already pinned and is immutable")
        self._rollback_target = target

    @property
    def rollback_target(self) -> RollbackTarget | None:
        return self._rollback_target

    def register(
        self,
        strategy_id: str,
        content_hash: str,
        *,
        change_kind: str = "initial",
        notes: str = "",
    ) -> StrategyVersionInfo:
        """Record a new version of one strategy (deduplicated by hash)."""
        history = self._history.setdefault(strategy_id, [])
        for entry in history:
            if entry.content_hash == content_hash and entry.change_kind == change_kind:
                return entry
        version = len(history) + 1
        info = StrategyVersionInfo(
            strategy_id=strategy_id, version=version, content_hash=content_hash,
            change_kind=change_kind, registered_at_utc=_utc_now(), notes=notes,
        )
        history.append(info)
        self._current[strategy_id] = info
        return info

    def current(self, strategy_id: str) -> StrategyVersionInfo | None:
        return self._current.get(strategy_id)

    def history(self, strategy_id: str) -> tuple[StrategyVersionInfo, ...]:
        return tuple(self._history.get(strategy_id, ()))

    def all_histories(self) -> dict[str, tuple[StrategyVersionInfo, ...]]:
        return {sid: tuple(entries) for sid, entries in self._history.items()}


__all__ = ["CHANGE_KINDS", "RollbackTarget", "StrategyVersionInfo", "Versioning"]
