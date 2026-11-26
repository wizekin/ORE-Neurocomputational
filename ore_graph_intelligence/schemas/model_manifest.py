"""Model manifest and split manifest schemas for reproducible graph-learning artifacts.

Every trained model is registered as a :class:`ModelManifest` with a weights hash, the exact
graph snapshot, the leakage-safe split manifest, configuration, metrics, limitations, and
resource needs (ORE Point 8 step 24). Split manifests record the leakage policy, hashes, and
seed so identical inputs reproduce identical artifacts (step 12).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any, Mapping


GRAPH_MODEL_CONTRACT = "ORE.GRAPH_MODEL/1.0.0"

SPLIT_POLICIES = frozenset({
    "random_edge", "entity", "relation", "time", "source", "graph_component",
})


def _canonical(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def build_weights_hash(weight_arrays: Mapping[str, Any]) -> str:
    """Hash trained parameter arrays deterministically, array order and contents included."""
    hasher = sha256()
    for name in sorted(weight_arrays):
        array = weight_arrays[name]
        hasher.update(name.encode("utf-8"))
        hasher.update(b"|")
        hasher.update(repr(getattr(array, "shape", None)).encode("utf-8"))
        hasher.update(b"|")
        hasher.update(getattr(array, "tobytes", lambda: bytes(str(array), "utf-8"))())
    return hasher.hexdigest()


@dataclass(frozen=True, slots=True)
class SplitManifest:
    """Leakage-aware dataset split manifest (ORE Point 8 step 12)."""

    policy: str
    seed: int
    train_triples: tuple[tuple[str, str, str], ...]
    validation_triples: tuple[tuple[str, str, str], ...]
    test_triples: tuple[tuple[str, str, str], ...]
    train_digest: str = ""
    validation_digest: str = ""
    test_digest: str = ""
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.policy not in SPLIT_POLICIES:
            raise ValueError(f"policy must be one of {sorted(SPLIT_POLICIES)}")
        if self.seed < 0:
            raise ValueError("seed cannot be negative")
        object.__setattr__(self, "train_digest", sha256(_canonical(
            {"policy": self.policy, "role": "train", "triples": [list(t) for t in self.train_triples]},
        ).encode("utf-8")).hexdigest())
        object.__setattr__(self, "validation_digest", sha256(_canonical(
            {"policy": self.policy, "role": "validation", "triples": [list(t) for t in self.validation_triples]},
        ).encode("utf-8")).hexdigest())
        object.__setattr__(self, "test_digest", sha256(_canonical(
            {"policy": self.policy, "role": "test", "triples": [list(t) for t in self.test_triples]},
        ).encode("utf-8")).hexdigest())
        all_triples = list(self.train_triples) + list(self.validation_triples) + list(self.test_triples)
        if len(all_triples) != len(set(all_triples)):
            raise ValueError("splits must be disjoint")
        overlap = set(self.train_triples) & set(self.validation_triples)
        if overlap:
            raise ValueError("train and validation splits overlap")

    def entity_disjoint(self) -> bool:
        """Return whether entity-holdout leakage policy holds between test and train."""
        train_entities = {triple[0] for triple in self.train_triples} | {triple[2] for triple in self.train_triples}
        test_entities = {triple[0] for triple in self.test_triples} | {triple[2] for triple in self.test_triples}
        return not (train_entities & test_entities)

    def as_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy, "seed": self.seed,
            "train": {"count": len(self.train_triples), "digest": self.train_digest},
            "validation": {"count": len(self.validation_triples), "digest": self.validation_digest},
            "test": {"count": len(self.test_triples), "digest": self.test_digest},
            "entity_disjoint": self.entity_disjoint(),
            "notes": list(self.notes),
        }


@dataclass(frozen=True, slots=True)
class ModelManifest:
    """Reproducible registration record for one trained graph-learning artifact."""

    model_id: str
    model_type: str
    version: str
    weights_hash: str
    snapshot_hash: str
    split_digest: str
    config: Mapping[str, Any] = field(default_factory=dict)
    metrics: Mapping[str, float] = field(default_factory=dict)
    limitations: tuple[str, ...] = ()
    dependencies: tuple[str, ...] = ()
    resource_needs: Mapping[str, float] = field(default_factory=dict)
    trained_at_utc: str = ""

    def __post_init__(self) -> None:
        for name in ("model_id", "model_type", "version", "trained_at_utc"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} must be non-empty")
        if len(self.weights_hash) != 64:
            raise ValueError("weights_hash must be a SHA-256 hex digest")
        if len(self.split_digest) != 64:
            raise ValueError("split_digest must be a SHA-256 hex digest")
        if not isinstance(self.config, Mapping) or not isinstance(self.metrics, Mapping):
            raise TypeError("config and metrics must be mappings")

    def identity_key(self) -> str:
        """Stable cache/registry key: model version plus weights provenance."""
        return f"{self.model_id}@{self.version}:{self.weights_hash[:12]}:{self.snapshot_hash[:12]}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id, "model_type": self.model_type, "version": self.version,
            "weights_hash": self.weights_hash, "snapshot_hash": self.snapshot_hash,
            "split_digest": self.split_digest, "config": dict(self.config),
            "metrics": dict(self.metrics), "limitations": list(self.limitations),
            "dependencies": list(self.dependencies), "resource_needs": dict(self.resource_needs),
            "trained_at_utc": self.trained_at_utc,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ModelManifest":
        data = dict(value)
        for name in ("config", "metrics", "resource_needs"):
            data[name] = dict(data.get(name) or {})
        for name in ("limitations", "dependencies"):
            data[name] = tuple(data.get(name) or ())
        return cls(**data)


@dataclass
class ModelArtifact:
    """A manifest paired with the in-memory trained model object it describes."""

    manifest: ModelManifest
    model: Any

    def close(self) -> None:
        self.model = None


__all__ = [
    "GRAPH_MODEL_CONTRACT", "ModelArtifact", "ModelManifest", "SplitManifest",
    "build_weights_hash", "SPLIT_POLICIES",
]
