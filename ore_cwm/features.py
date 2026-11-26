"""Versioned compact cognitive-state feature extraction from authoritative snapshots.

Features are a predictive projection of the shared workspace, never a replacement state
representation: source objects stay in the workspace and every value is labeled observed,
deterministically derived, or predicted.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ore_workspace.goals import GoalEvaluator
from ore_workspace.models import WorkspaceSnapshot, content_fingerprint

from .contracts import (
    CWM_FEATURE_SCHEMA_VERSION,
    MISSING,
    CognitiveStateFeatures,
    FeatureProvenance,
    FeatureValue,
    MissingMeasurement,
    ObservedOutcome,
    ResourceState,
)

EVIDENCE_FIELDS = (
    "lexical_hypotheses",
    "syntactic_hypotheses",
    "semantic_hypotheses",
    "discourse_context",
    "retrieved_memories",
    "artifact_references",
)

FEATURE_NAMES = (
    "goal_completion",
    "evidence_coverage",
    "hypothesis_count",
    "dependency_backlog",
    "open_questions",
    "unresolved_conflicts",
    "recent_module_success",
    "resource_pressure",
)

DEFAULT_FEATURE_DEFAULTS: dict[str, float] = {
    "goal_completion": 0.0,
    "evidence_coverage": 0.0,
    "hypothesis_count": 0.0,
    "dependency_backlog": 0.0,
    "open_questions": 0.0,
    "unresolved_conflicts": 1.0,
    "recent_module_success": 0.0,
    "resource_pressure": 1.0,
}


def _field_present(state: Mapping[str, Any], field_name: str) -> bool:
    """Report whether one evidence-bearing workspace field currently holds content."""
    return bool(state.get(field_name))


class FeatureExtractor:
    """Extract the versioned CWM feature vector from one consistent workspace snapshot."""

    def __init__(self, goal_evaluator: GoalEvaluator | None = None, recent_window: int = 8) -> None:
        """Reuse the existing goal evaluator so CWM never competes with its goal score."""
        if recent_window < 1:
            raise ValueError("recent_window must be a positive integer")
        self.goal_evaluator = goal_evaluator or GoalEvaluator()
        self.recent_window = int(recent_window)

    def extract(
        self,
        snapshot: WorkspaceSnapshot,
        *,
        resource_state: ResourceState | None = None,
        unresolved_conflicts: int | None = None,
        recent_executions: Sequence[ObservedOutcome] = (),
    ) -> CognitiveStateFeatures:
        """Derive the compact feature vector tied to the exact snapshot revision."""
        evaluation = self.goal_evaluator.evaluate(snapshot)
        state = snapshot.state
        evidence_present = sum(
            1 for name in EVIDENCE_FIELDS if _field_present(state, name))
        evidence_coverage = evidence_present / len(EVIDENCE_FIELDS)
        hypothesis_count = (
            len(state.get("lexical_hypotheses", {}) or {})
            + len(state.get("syntactic_hypotheses", []) or [])
            + len(state.get("semantic_hypotheses", []) or [])
        )
        pending = state.get("pending_updates", []) or []
        open_questions = len(pending) + _count_open_questions(state)
        pressure = (
            resource_state.resource_pressure()
            if resource_state is not None else MISSING
        )
        values = {
            "goal_completion": FeatureValue(
                "goal_completion", float(evaluation.score), FeatureProvenance.DERIVED,
                "GoalEvaluator weighted score at this revision (Step 18 reuses its criteria)."),
            "evidence_coverage": FeatureValue(
                "evidence_coverage", evidence_coverage, FeatureProvenance.DERIVED,
                "Fraction of evidence-bearing workspace fields holding content."),
            "hypothesis_count": FeatureValue(
                "hypothesis_count", float(hypothesis_count), FeatureProvenance.OBSERVED,
                "Count of lexical, syntactic, and semantic hypotheses in the snapshot."),
            "dependency_backlog": FeatureValue(
                "dependency_backlog", float(len(pending)), FeatureProvenance.OBSERVED,
                "Count of deferred findings awaiting dependency resolution."),
            "open_questions": FeatureValue(
                "open_questions", float(open_questions), FeatureProvenance.OBSERVED,
                "Unresolved questions plus deferred updates at this revision."),
            "unresolved_conflicts": FeatureValue(
                "unresolved_conflicts",
                float(unresolved_conflicts) if unresolved_conflicts is not None else MISSING,
                FeatureProvenance.OBSERVED,
                "Unresolved ConflictRegistry entries when a registry count is supplied."),
            "recent_module_success": FeatureValue(
                "recent_module_success", _recent_success(recent_executions[-self.recent_window:]),
                FeatureProvenance.DERIVED,
                "Share of recent executed outcomes whose artifacts were accepted."),
            "resource_pressure": FeatureValue(
                "resource_pressure", pressure, FeatureProvenance.OBSERVED,
                "Memory utilization or CPU load from the supplied resource measurements."),
        }
        feature_snapshot_hash = content_fingerprint({
            "schema_version": CWM_FEATURE_SCHEMA_VERSION,
            "workspace_id": snapshot.workspace_id,
            "workspace_revision": snapshot.revision,
            "goal": str(state.get("current_goal", "")),
            "values": {
                name: (None if item.value is MISSING else float(item.value))
                for name, item in values.items()
            },
        })
        return CognitiveStateFeatures(
            schema_version=CWM_FEATURE_SCHEMA_VERSION,
            workspace_id=snapshot.workspace_id,
            workspace_revision=snapshot.revision,
            goal=str(state.get("current_goal", "")),
            feature_snapshot_hash=feature_snapshot_hash,
            values=values,
        )


def _count_open_questions(state: Mapping[str, Any]) -> int:
    """Count unresolved question records that the snapshot reports as open."""
    questions = state.get("unresolved_questions", []) or []
    return sum(
        1 for item in questions
        if isinstance(item, Mapping) and not bool(item.get("resolved", False))
    )


def _recent_success(executions: Sequence[ObservedOutcome]) -> float | MissingMeasurement:
    """Return the accepted-artifact share of recent executions, or missing when unsupported."""
    candidates = [
        item for item in executions
        if item.execution_status.value == "executed"
        and item.artifact_lifecycle.value in ("accepted", "generated")
    ]
    if not candidates:
        return MISSING
    accepted = sum(1 for item in candidates if item.artifact_lifecycle.value == "accepted")
    return accepted / len(candidates)


__all__ = [
    "DEFAULT_FEATURE_DEFAULTS",
    "EVIDENCE_FIELDS",
    "FEATURE_NAMES",
    "FeatureExtractor",
]
