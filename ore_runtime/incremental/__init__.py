"""Incremental computation: dependency graphs and invalidation planning."""
from .dependency_graph import (
    ArtifactRecord, ArtifactState, DependencyGraph, DependencyRef, SourceRecord,
    compute_artifact_hash,
)
from .invalidation import InvalidationEngine, InvalidationReport

__all__ = [
    "ArtifactRecord", "ArtifactState", "DependencyGraph", "DependencyRef",
    "InvalidationEngine", "InvalidationReport", "SourceRecord",
    "compute_artifact_hash",
]
