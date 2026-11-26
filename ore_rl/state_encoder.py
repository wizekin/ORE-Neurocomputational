"""Deterministic, versioned state encoder for ore_rl (Phase B, Steps 7-14).

Builds the RL observation from typed ORE signals: task context, goal progress,
workspace/evidence state, capability state, dependency state, resource state,
and uncertainty/history features. The transformation is deterministic and its
version, feature names, and normalization statistics are persisted with the model.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from .contracts import OREState, RL_SCHEMA_VERSION, stable_hash

ENCODER_FEATURE_GROUPS: tuple[str, ...] = (
    "task_context",
    "goal_progress",
    "workspace_evidence",
    "capability_state",
    "dependency_state",
    "resource_state",
    "uncertainty_history",
)


@dataclass(frozen=True, slots=True)
class StateEncoderVersion:
    """Explicit encoder identity persisted with every state and trained model (Step 14)."""

    version: str = "1.0.0"
    feature_names: tuple[str, ...] = ()

    @property
    def fingerprint(self) -> str:
        return stable_hash({"version": self.version, "feature_names": list(self.feature_names)})


@dataclass(frozen=True, slots=True)
class TaskContextSignal:
    """Raw task-context inputs (Step 7). Categorical labels stay canonical for audit."""

    task_family: str
    domain: str
    requested_output: str
    difficulty: float
    urgency: float
    constraints: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class GoalProgressSignal:
    """Typed goal-progress counters instead of one opaque score (Step 8)."""

    completed_subgoals: tuple[str, ...] = ()
    remaining_subgoals: tuple[str, ...] = ()
    unresolved_requirements: tuple[str, ...] = ()
    completion_confidence: float = 0.0


@dataclass(frozen=True, slots=True)
class WorkspaceEvidenceSignal:
    """Workspace/evidence summary bound to a workspace revision (Step 9)."""

    workspace_revision: int
    finding_count: int
    verified_finding_count: int
    unverified_finding_count: int
    contradiction_count: int
    mean_source_reliability: float
    evidence_quality_score: float
    evidence_quantity_score: float


@dataclass(frozen=True, slots=True)
class CapabilityStateSignal:
    """Per-module readiness, health, and estimated cost (Step 10)."""

    module_ids: tuple[str, ...]
    readiness: tuple[float, ...]
    health: tuple[float, ...]
    queue_lengths: tuple[int, ...]
    estimated_cost: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class DependencyStateSignal:
    """Dependency mask inputs: m_a(s) = 1 iff all prerequisites are complete (Step 11)."""

    action_ids: tuple[str, ...]
    dependency_satisfied: tuple[bool, ...]


@dataclass(frozen=True, slots=True)
class ResourceStateSignal:
    """Measured resource state, kept separate from forecasts (Step 12)."""

    remaining_wall_clock_fraction: float
    remaining_cpu_fraction: float
    remaining_memory_fraction: float
    remaining_module_calls_fraction: float
    reserved_wall_clock_fraction: float


@dataclass(frozen=True, slots=True)
class UncertaintyHistorySignal:
    """Calibrated uncertainty and per-action history; missing history stays explicit (Step 13)."""

    evaluator_confidence: float
    action_sample_counts: Mapping[str, int]
    action_mean_outcomes: Mapping[str, float]
    similar_task_success_rate: float | None = None
    similar_task_sample_count: int = 0


@dataclass(frozen=True, slots=True)
class RawStateSignals:
    """Bundle of typed raw signals produced by the environment or coordinator adapter."""

    task: TaskContextSignal
    goal: GoalProgressSignal
    workspace: WorkspaceEvidenceSignal
    capabilities: CapabilityStateSignal
    dependencies: DependencyStateSignal
    resources: ResourceStateSignal
    uncertainty: UncertaintyHistorySignal
    task_id: str = ""
    episode_id: str = ""
    step_index: int = 0
    provenance: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class StateSpaceSpec:
    """Declared feature layout of the encoded state vector."""

    names: tuple[str, ...]
    encoder_version: str

    @property
    def dim(self) -> int:
        return len(self.names)


@dataclass(slots=True)
class NormalizationStats:
    """Persisted (mu, sigma) computed from training data only (Step 7)."""

    feature_names: tuple[str, ...]
    mu: np.ndarray
    sigma: np.ndarray
    epsilon: float = 1e-8

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mu) / (self.sigma + self.epsilon)

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_names": list(self.feature_names),
            "mu": self.mu.tolist(),
            "sigma": self.sigma.tolist(),
            "epsilon": self.epsilon,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> NormalizationStats:
        return cls(
            feature_names=tuple(data["feature_names"]),
            mu=np.asarray(data["mu"], dtype=np.float64),
            sigma=np.asarray(data["sigma"], dtype=np.float64),
            epsilon=float(data.get("epsilon", 1e-8)),
        )

    @classmethod
    def fit(cls, names: Sequence[str], rows: Sequence[Mapping[str, float]]) -> NormalizationStats:
        matrix = np.asarray(
            [[float(row.get(name, math.nan)) for name in names] for row in rows],
            dtype=np.float64,
        )
        with np.errstate(invalid="ignore"):
            mu = np.nanmean(matrix, axis=0) if len(rows) else np.zeros(len(names))
            sigma = np.nanstd(matrix, axis=0) if len(rows) else np.ones(len(names))
        mu = np.nan_to_num(mu, nan=0.0)
        sigma = np.nan_to_num(sigma, nan=1.0)
        sigma = np.where(sigma < 1e-8, 1.0, sigma)
        return cls(feature_names=tuple(names), mu=mu, sigma=sigma)


class StateEncoder:
    """Deterministic transformation from typed signals to the state vector (Step 14).

    ``encode`` is a pure function of the signals: identical inputs always produce an
    identical feature mapping and state id, which is asserted by unit tests. The
    encoder version, feature names, and any normalization statistics must be
    persisted next to trained policies.
    """

    def __init__(self, *, task_families: Sequence[str] = (), domains: Sequence[str] = ()) -> None:
        self.task_families: tuple[str, ...] = tuple(task_families)
        self.domains: tuple[str, ...] = tuple(domains)
        self._version = StateEncoderVersion(version="1.0.0")
        self._normalization: NormalizationStats | None = None

    # -- configuration -----------------------------------------------------
    @property
    def version(self) -> StateEncoderVersion:
        return self._version

    @property
    def normalization(self) -> NormalizationStats | None:
        return self._normalization

    def set_normalization(self, stats: NormalizationStats) -> None:
        if tuple(stats.feature_names) != self.feature_names:
            raise ValueError("normalization feature names do not match encoder layout")
        self._normalization = stats

    def fit_normalization(self, signals: Sequence[RawStateSignals]) -> NormalizationStats:
        rows = [self._raw_features(item) for item in signals]
        stats = NormalizationStats.fit(self.feature_names, rows)
        self._normalization = stats
        return stats

    # -- feature layout ----------------------------------------------------
    @property
    def feature_names(self) -> tuple[str, ...]:
        names: list[str] = []
        names.extend(self._task_feature_names())
        names.extend(self._goal_feature_names())
        names.extend(self._workspace_feature_names())
        names.extend(self._capability_feature_names())
        names.extend(self._dependency_feature_names())
        names.extend(self._resource_feature_names())
        names.extend(self._uncertainty_feature_names())
        return tuple(names)

    def spec(self) -> StateSpaceSpec:
        return StateSpaceSpec(names=self.feature_names, encoder_version=self._version.version)

    def _task_feature_names(self) -> list[str]:
        names = [f"task_family::{item}" for item in self.task_families]
        names += [f"domain::{item}" for item in self.domains]
        names += ["task.difficulty", "task.urgency", "task.constraint_count"]
        return names

    def _goal_feature_names(self) -> list[str]:
        return [
            "goal.completed_count",
            "goal.remaining_count",
            "goal.unresolved_count",
            "goal.completion_confidence",
        ]

    def _workspace_feature_names(self) -> list[str]:
        return [
            "workspace.revision",
            "workspace.finding_count",
            "workspace.verified_fraction",
            "workspace.contradiction_count",
            "workspace.mean_source_reliability",
            "workspace.evidence_quality",
            "workspace.evidence_quantity",
        ]

    def _capability_feature_names(self) -> list[str]:
        return [
            "capability.mean_readiness",
            "capability.mean_health",
            "capability.total_queue",
            "capability.mean_estimated_cost",
        ]

    def _dependency_feature_names(self) -> list[str]:
        return [
            "dependency.satisfied_fraction",
            "dependency.eligible_count",
        ]

    def _resource_feature_names(self) -> list[str]:
        return [
            "resource.remaining_wall_clock",
            "resource.remaining_cpu",
            "resource.remaining_memory",
            "resource.remaining_module_calls",
            "resource.reserved_wall_clock",
        ]

    def _uncertainty_feature_names(self) -> list[str]:
        return [
            "uncertainty.evaluator_confidence",
            "uncertainty.history_sample_total",
            "uncertainty.history_mean_outcome",
            "uncertainty.similar_task_success",
            "uncertainty.similar_task_available",
        ]

    # -- encoding ----------------------------------------------------------
    def _raw_features(self, signals: RawStateSignals) -> dict[str, float]:
        features: dict[str, float] = {}
        task = signals.task
        for family in self.task_families:
            features[f"task_family::{family}"] = 1.0 if task.task_family == family else 0.0
        for domain in self.domains:
            features[f"domain::{domain}"] = 1.0 if task.domain == domain else 0.0
        features["task.difficulty"] = float(task.difficulty)
        features["task.urgency"] = float(task.urgency)
        features["task.constraint_count"] = float(len(task.constraints))

        goal = signals.goal
        features["goal.completed_count"] = float(len(goal.completed_subgoals))
        features["goal.remaining_count"] = float(len(goal.remaining_subgoals))
        features["goal.unresolved_count"] = float(len(goal.unresolved_requirements))
        features["goal.completion_confidence"] = float(goal.completion_confidence)

        workspace = signals.workspace
        total = max(1, workspace.finding_count)
        features["workspace.revision"] = float(workspace.workspace_revision)
        features["workspace.finding_count"] = float(workspace.finding_count)
        features["workspace.verified_fraction"] = (
            workspace.verified_finding_count / total if workspace.finding_count else 0.0
        )
        features["workspace.contradiction_count"] = float(workspace.contradiction_count)
        features["workspace.mean_source_reliability"] = float(workspace.mean_source_reliability)
        features["workspace.evidence_quality"] = float(workspace.evidence_quality_score)
        features["workspace.evidence_quantity"] = float(workspace.evidence_quantity_score)

        capabilities = signals.capabilities
        n_modules = len(capabilities.module_ids)
        features["capability.mean_readiness"] = (
            sum(capabilities.readiness) / n_modules if n_modules else 0.0
        )
        features["capability.mean_health"] = (
            sum(capabilities.health) / n_modules if n_modules else 0.0
        )
        features["capability.total_queue"] = float(sum(capabilities.queue_lengths))
        features["capability.mean_estimated_cost"] = (
            sum(capabilities.estimated_cost) / n_modules if n_modules else 0.0
        )

        dependencies = signals.dependencies
        n_actions = len(dependencies.action_ids)
        features["dependency.satisfied_fraction"] = (
            sum(1.0 for flag in dependencies.dependency_satisfied if flag) / n_actions
            if n_actions
            else 0.0
        )
        features["dependency.eligible_count"] = float(
            sum(1 for flag in dependencies.dependency_satisfied if flag)
        )

        resources = signals.resources
        features["resource.remaining_wall_clock"] = float(resources.remaining_wall_clock_fraction)
        features["resource.remaining_cpu"] = float(resources.remaining_cpu_fraction)
        features["resource.remaining_memory"] = float(resources.remaining_memory_fraction)
        features["resource.remaining_module_calls"] = float(
            resources.remaining_module_calls_fraction
        )
        features["resource.reserved_wall_clock"] = float(resources.reserved_wall_clock_fraction)

        uncertainty = signals.uncertainty
        sample_total = sum(uncertainty.action_sample_counts.values())
        mean_outcome = (
            sum(uncertainty.action_mean_outcomes.values())
            / max(1, len(uncertainty.action_mean_outcomes))
        )
        features["uncertainty.evaluator_confidence"] = float(uncertainty.evaluator_confidence)
        features["uncertainty.history_sample_total"] = float(sample_total)
        features["uncertainty.history_mean_outcome"] = float(mean_outcome)
        features["uncertainty.similar_task_success"] = (
            float(uncertainty.similar_task_success_rate)
            if uncertainty.similar_task_success_rate is not None
            else 0.0
        )
        features["uncertainty.similar_task_available"] = (
            1.0 if uncertainty.similar_task_success_rate is not None else 0.0
        )
        return features

    def encode(self, signals: RawStateSignals) -> OREState:
        """Produce the typed OREState with an explicit missing-feature list."""
        raw = self._raw_features(signals)
        names = self.feature_names
        features: dict[str, float] = {}
        missing: list[str] = []
        for name in names:
            if name in raw:
                features[name] = float(raw[name])
            else:
                missing.append(name)
        if self._normalization is not None:
            vector = np.asarray([features.get(name, math.nan) for name in names], dtype=np.float64)
            normalized = np.where(
                np.isnan(vector), math.nan, self._normalization.transform(vector)
            )
            features = {name: float(normalized[i]) for i, name in enumerate(names)}
        state_id = "state_" + stable_hash(
            {
                "task_id": signals.task.task_id if hasattr(signals.task, "task_id") else "",
                "raw_task_id": signals.task_id,
                "episode_id": signals.episode_id,
                "step_index": signals.step_index,
                "features": {name: features.get(name) for name in names},
                "missing": missing,
                "encoder": self._version.version,
            }
        )[:24]
        return OREState(
            schema_version=RL_SCHEMA_VERSION,
            state_id=state_id,
            task_id=signals.task_id,
            task_family=signals.task.task_family,
            workspace_id=str(signals.provenance.get("workspace_id", "")),
            workspace_revision=signals.workspace.workspace_revision,
            episode_id=signals.episode_id,
            step_index=signals.step_index,
            encoder_version=self._version.version,
            feature_names=names,
            features=features,
            missing_features=tuple(missing),
            provenance={
                "encoder_fingerprint": self._version.fingerprint,
                **dict(signals.provenance),
            },
        )

    def to_vector(self, state: OREState) -> np.ndarray:
        """Numeric vector in declared feature order; NaN marks explicit missingness."""
        return np.asarray(
            [state.features.get(name, math.nan) for name in self.feature_names],
            dtype=np.float64,
        )

    def config_summary(self) -> dict[str, Any]:
        return {
            "encoder_version": self._version.version,
            "feature_names": list(self.feature_names),
            "task_families": list(self.task_families),
            "domains": list(self.domains),
            "normalization": self._normalization.to_dict() if self._normalization else None,
        }
