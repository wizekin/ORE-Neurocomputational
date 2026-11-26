"""Durable consumer deduplication keyed by (consumer_id, event_id)."""
from __future__ import annotations

import json
import threading
from collections.abc import Callable
from enum import Enum
from pathlib import Path
from typing import Any


class ApplyStatus(str, Enum):
    """Report whether an idempotent effect was applied or already recorded."""

    APPLIED = "applied"
    ALREADY_APPLIED = "already_applied"


class IdempotencyStore:
    """Keep durable processed-event keys so redelivery cannot duplicate effects."""

    def __init__(self, persistence_path: str | Path | None = None) -> None:
        """Initialize the dedup table and load any persisted keys from disk."""
        self._lock = threading.RLock()
        self._processed: dict[tuple[str, str], str] = {}
        self._path = Path(persistence_path).expanduser() if persistence_path else None
        self._duplicate_deliveries = 0
        if self._path is not None and self._path.exists():
            self._load()

    def _load(self) -> None:
        """Load previously processed keys from the durable JSON snapshot."""
        if self._path is None:
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        for row in raw.get("processed", []):
            self._processed[(row["consumer_id"], row["event_id"])] = row.get(
                "applied_at", ""
            )

    def _persist(self) -> None:
        """Persist the processed-key table to disk inside the caller's lock."""
        if self._path is None:
            return
        rows = [
            {"consumer_id": consumer, "event_id": event_id, "applied_at": applied_at}
            for (consumer, event_id), applied_at in self._processed.items()
        ]
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps({"processed": rows}, sort_keys=True, indent=1), encoding="utf-8",
        )

    @staticmethod
    def _key(consumer_id: str, event_id: str) -> tuple[str, str]:
        """Build the durable deduplication key for one consumer and event."""
        if not consumer_id.strip() or not event_id.strip():
            raise ValueError("consumer_id and event_id must be non-empty.")
        return consumer_id, event_id

    def seen(self, consumer_id: str, event_id: str) -> bool:
        """Check whether a consumer already processed one event identifier."""
        with self._lock:
            return self._key(consumer_id, event_id) in self._processed

    def apply_once(
        self, consumer_id: str, event_id: str,
        effect: Callable[[], Any],
    ) -> tuple[ApplyStatus, Any]:
        """Apply one effect exactly once per (consumer_id, event_id) key.

        The processed marker and the effect are committed in the same critical
        section, so a crash between them cannot leave a durable marker without
        the applied logical effect or vice versa.
        """
        with self._lock:
            key = self._key(consumer_id, event_id)
            if key in self._processed:
                self._duplicate_deliveries += 1
                return ApplyStatus.ALREADY_APPLIED, None
            from ..contracts import utc_now
            applied_at = utc_now()
            result = effect()
            self._processed[key] = applied_at
            self._persist()
            return ApplyStatus.APPLIED, result

    def mark_processed(self, consumer_id: str, event_id: str) -> ApplyStatus:
        """Record a processed key without an effect for externally applied work."""
        with self._lock:
            key = self._key(consumer_id, event_id)
            if key in self._processed:
                self._duplicate_deliveries += 1
                return ApplyStatus.ALREADY_APPLIED
            from ..contracts import utc_now
            self._processed[key] = utc_now()
            self._persist()
            return ApplyStatus.APPLIED

    def forget(self, consumer_id: str, event_id: str) -> None:
        """Drop one processed key for tests and explicit replay resets."""
        with self._lock:
            self._processed.pop(self._key(consumer_id, event_id), None)
            self._persist()

    def stats(self) -> dict[str, Any]:
        """Return dedup counters for the metrics report."""
        with self._lock:
            return {
                "processed_keys": len(self._processed),
                "duplicate_deliveries": self._duplicate_deliveries,
                "persistence_path": str(self._path) if self._path else None,
            }
