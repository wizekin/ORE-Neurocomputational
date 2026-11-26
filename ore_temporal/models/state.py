"""Model-state isolation, checkpointing, and expiration (specification Steps 17 and 23).

State is separated by source, task, model version, and security boundary.  Restoration
happens only when model version, schema, source identity, and state timestamp are
compatible; expired or incompatible checkpoints are rejected rather than reinterpreted.
"""
from __future__ import annotations

from dataclasses import dataclass
import time as _time
from typing import Any, Iterable, Mapping

import numpy as np

__all__ = ("ModelStateRecord", "ModelStateStore", "StateCompatibilityError")


class StateCompatibilityError(ValueError):
    """Raised when a checkpoint is incompatible with the requesting stream or model."""


@dataclass(frozen=True, slots=True)
class ModelStateRecord:
    """One checkpointed model state with full compatibility metadata."""

    task_id: str
    source_id: str
    model_family: str
    model_version: str
    config_hash: str
    channels: tuple[str, ...]
    state: tuple[float, ...]
    state_timestamp: float
    created_at: float
    schema_version: str = "ore_temporal-state-1"

    @property
    def identity(self) -> tuple[str, str, str, str]:
        """Return the (task, source, family, version) isolation key."""
        return (self.task_id, self.source_id, self.model_family, self.model_version)

    def nbytes(self) -> int:
        """Return the byte size of the stored state."""
        return len(self.state) * 8

    def to_dict(self) -> dict[str, Any]:
        """Serialize the checkpoint record."""
        return {
            "task_id": self.task_id,
            "source_id": self.source_id,
            "model_family": self.model_family,
            "model_version": self.model_version,
            "config_hash": self.config_hash,
            "channels": list(self.channels),
            "state_size": len(self.state),
            "state_timestamp": self.state_timestamp,
            "created_at": self.created_at,
            "schema_version": self.schema_version,
        }


class ModelStateStore:
    """An in-memory, size-bounded state store with per-source isolation and expiration."""

    def __init__(self, size_limit_bytes: int = 1_048_576,
                 retention_seconds: float = 3600.0,
                 clock: Any = _time.monotonic) -> None:
        """Configure the byte budget, retention window, and injectable clock."""
        if size_limit_bytes <= 0:
            raise ValueError("size_limit_bytes must be positive")
        if retention_seconds < 0.0:
            raise ValueError("retention_seconds must be non-negative")
        self.size_limit_bytes = int(size_limit_bytes)
        self.retention_seconds = float(retention_seconds)
        self._clock = clock
        self._records: dict[tuple[str, str, str, str], ModelStateRecord] = {}

    def checkpoint(self, record: ModelStateRecord, *, authorized: bool = True) -> None:
        """Store a checkpoint only when explicitly authorized (Step 50)."""
        if not authorized:
            raise PermissionError(
                "state checkpointing requires explicit authorization; "
                "unauthorized continuation state is not persisted")
        self._records[record.identity] = record
        self._enforce_size_limit()

    def restore(self, task_id: str, source_id: str, model_family: str,
                model_version: str, *, channels: tuple[str, ...], config_hash: str,
                current_timestamp: float) -> ModelStateRecord:
        """Restore a compatible checkpoint or raise StateCompatibilityError."""
        key = (task_id, source_id, model_family, model_version)
        record = self._records.get(key)
        if record is None:
            raise StateCompatibilityError(
                f"no checkpoint stored for task/source/family/version {key}")
        if record.config_hash != config_hash:
            raise StateCompatibilityError(
                "checkpoint config hash does not match the requesting model configuration")
        if tuple(record.channels) != tuple(channels):
            raise StateCompatibilityError("checkpoint schema does not match the request schema")
        if record.state_timestamp > current_timestamp + 1e-9:
            raise StateCompatibilityError(
                "checkpoint state timestamp lies in the future relative to the window")
        if self._expired(record):
            del self._records[key]
            raise StateCompatibilityError("checkpoint expired under the retention policy")
        return record

    def has_checkpoint(self, task_id: str, source_id: str, model_family: str,
                       model_version: str) -> bool:
        """Report whether a checkpoint exists for the isolation key."""
        return (task_id, source_id, model_family, model_version) in self._records

    def reset(self, task_id: str, source_id: str, model_family: str | None = None,
              model_version: str | None = None) -> int:
        """Explicitly reset (drop) state for a stream; returns the number of records dropped."""
        dropped = 0
        for key in list(self._records):
            if key[0] == task_id and key[1] == source_id and (
                    model_family is None or key[2] == model_family) and (
                    model_version is None or key[3] == model_version):
                del self._records[key]
                dropped += 1
        return dropped

    def expire(self, now: float | None = None) -> list[dict[str, Any]]:
        """Drop expired records under the retention policy and report what was expired."""
        current = float(now if now is not None else self._clock())
        expired: list[dict[str, Any]] = []
        for key, record in list(self._records.items()):
            if current - record.created_at > self.retention_seconds:
                expired.append(record.to_dict())
                del self._records[key]
        return expired

    def total_bytes(self) -> int:
        """Return the total bytes currently stored."""
        return sum(record.nbytes() for record in self._records.values())

    def snapshot_keys(self) -> tuple[tuple[str, str, str, str], ...]:
        """Return the active isolation keys (never cross-stream contamination)."""
        return tuple(self._records)

    def summary(self) -> Mapping[str, Any]:
        """Return a store summary for diagnostics."""
        return {
            "records": len(self._records),
            "total_bytes": self.total_bytes(),
            "size_limit_bytes": self.size_limit_bytes,
            "retention_seconds": self.retention_seconds,
            "keys": [list(key) for key in self._records],
        }

    def _expired(self, record: ModelStateRecord) -> bool:
        """Report whether a record has exceeded its retention window."""
        return (float(self._clock()) - record.created_at) > self.retention_seconds

    def _enforce_size_limit(self) -> None:
        """Evict the oldest records until the byte budget is respected."""
        while self.total_bytes() > self.size_limit_bytes and self._records:
            oldest_key = min(self._records, key=lambda key: self._records[key].created_at)
            del self._records[oldest_key]


def record_from_array(task_id: str, source_id: str, model_family: str, model_version: str,
                      config_hash: str, channels: tuple[str, ...], state: np.ndarray,
                      state_timestamp: float, created_at: float | None = None,
                      ) -> ModelStateRecord:
    """Build a checkpoint record from a model state array."""
    array = np.asarray(state, dtype=float).reshape(-1)
    return ModelStateRecord(
        task_id=task_id,
        source_id=source_id,
        model_family=model_family,
        model_version=model_version,
        config_hash=config_hash,
        channels=tuple(channels),
        state=tuple(float(item) for item in array),
        state_timestamp=float(state_timestamp),
        created_at=float(created_at if created_at is not None else _time.monotonic()),
    )


def records_for_source(records: Iterable[ModelStateRecord],
                       source_id: str) -> tuple[ModelStateRecord, ...]:
    """Filter checkpoint records by source to verify isolation."""
    return tuple(record for record in records if record.source_id == source_id)
