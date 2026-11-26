"""Content-addressed strategy artifact store (registry layer).

Artifacts carry an immutable manifest (benchmark identity, split digests, metrics
summary, evaluator version, lineage, created-at) and are stored content-addressed:
the store recomputes the payload hash on save and verifies it on load, so tampered or
unmanifested artifacts are rejected (acceptance tests 23, 24, 28).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from ..domain.genome import content_digest


@dataclass(frozen=True, slots=True)
class StrategyManifest:
    """Immutable artifact manifest (acceptance test 28: manifest required)."""

    benchmark_id: str
    benchmark_version: str
    split_digest: str
    evaluator_version: str
    metrics_summary: Mapping[str, Any] = field(default_factory=dict)
    run_id: str | None = None
    created_at_utc: str = ""
    source_version: str = ""
    fitness_config: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.benchmark_id or not self.benchmark_version:
            raise ValueError("manifest must carry the benchmark identity and version")
        if not self.evaluator_version:
            raise ValueError("manifest must carry the evaluator version")
        if not self.split_digest:
            raise ValueError("manifest must carry the split digest")

    def as_dict(self) -> dict[str, Any]:
        return {
            "benchmark_id": self.benchmark_id,
            "benchmark_version": self.benchmark_version,
            "split_digest": self.split_digest,
            "evaluator_version": self.evaluator_version,
            "metrics_summary": dict(self.metrics_summary),
            "run_id": self.run_id,
            "created_at_utc": self.created_at_utc,
            "source_version": self.source_version,
            "fitness_config": dict(self.fitness_config),
        }

    def digest(self) -> str:
        return content_digest(self.as_dict())


@dataclass(frozen=True, slots=True)
class StrategyArtifact:
    """One immutable strategy artifact: payload + manifest + verified hash."""

    payload: Mapping[str, Any]
    manifest: StrategyManifest
    content_hash: str

    def __post_init__(self) -> None:
        if not self.manifest:
            raise ValueError("artifact requires a manifest")
        recomputed = content_digest(dict(self.payload))
        if recomputed != self.content_hash:
            raise ValueError(
                "artifact content hash mismatch: payload was modified after hashing"
            )

    @classmethod
    def build(
        cls, payload: Mapping[str, Any], manifest: StrategyManifest,
    ) -> "StrategyArtifact":
        """Build an artifact with a computed, verified content hash."""
        return cls(payload=dict(payload), manifest=manifest, content_hash=content_digest(dict(payload)))

    def as_dict(self) -> dict[str, Any]:
        return {
            "content_hash": self.content_hash,
            "payload": dict(self.payload),
            "manifest": self.manifest.as_dict(),
        }


class ArtifactStore:
    """In-memory content-addressed artifact store with integrity verification."""

    def __init__(self) -> None:
        self._artifacts: dict[str, StrategyArtifact] = {}

    def save(self, artifact: StrategyArtifact) -> str:
        """Store one artifact under its content hash."""
        if artifact.manifest is None:
            raise ValueError("artifact without a manifest is rejected")
        self._artifacts[artifact.content_hash] = artifact
        return artifact.content_hash

    def load(self, content_hash: str) -> StrategyArtifact:
        """Load and verify one artifact by content hash."""
        if content_hash not in self._artifacts:
            raise KeyError(f"unknown artifact hash {content_hash!r}")
        artifact = self._artifacts[content_hash]
        if content_digest(dict(artifact.payload)) != artifact.content_hash:
            raise ValueError(f"artifact {content_hash!r} failed integrity verification")
        return artifact

    def has(self, content_hash: str) -> bool:
        return content_hash in self._artifacts

    def __len__(self) -> int:
        return len(self._artifacts)

    def hashes(self) -> list[str]:
        return sorted(self._artifacts)


__all__ = ["ArtifactStore", "StrategyArtifact", "StrategyManifest"]
