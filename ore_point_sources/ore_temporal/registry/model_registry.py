"""Model artifact registry, lifecycle, and promotion gates (Steps 46-47, 50, Section 11)."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

__all__ = ("LifecycleState", "ModelArtifactRecord", "PromotionDecision",
           "TemporalModelRegistry", "evaluate_promotion")


class LifecycleState(str, Enum):
    """Lifecycle states for model artifacts in the registry."""

    PROPOSED = "proposed"
    VALIDATED = "validated"
    APPROVED = "approved"
    DEPRECATED = "deprecated"


@dataclass(frozen=True, slots=True)
class ModelArtifactRecord:
    """Step 50 -- registry metadata for one model artifact."""

    artifact_hash: str
    family: str
    architecture: str
    dataset_lineage: tuple[str, ...]
    training_config: Mapping[str, Any]
    validation_metrics: Mapping[str, float]
    license: str
    limitations: tuple[str, ...]
    approval_state: LifecycleState = LifecycleState.PROPOSED
    trained_artifact_immutable: bool = True

    def __post_init__(self) -> None:
        """Validate the artifact record."""
        if not self.artifact_hash:
            raise ValueError("artifact_hash must be non-empty")
        if not self.family or not self.architecture:
            raise ValueError("family and architecture must be non-empty")
        if not isinstance(self.approval_state, LifecycleState):
            raise ValueError("approval_state must be a LifecycleState")

    def to_dict(self) -> dict[str, Any]:
        """Serialize the artifact record."""
        return {
            "artifact_hash": self.artifact_hash,
            "family": self.family,
            "architecture": self.architecture,
            "dataset_lineage": list(self.dataset_lineage),
            "training_config": dict(self.training_config),
            "validation_metrics": dict(self.validation_metrics),
            "license": self.license,
            "limitations": list(self.limitations),
            "approval_state": self.approval_state.value,
            "trained_artifact_immutable": self.trained_artifact_immutable,
        }


@dataclass(frozen=True, slots=True)
class PromotionDecision:
    """The outcome of evaluating the promotion gates (Section 11)."""

    promoted: bool
    failed_gates: tuple[str, ...]
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Serialize the promotion decision."""
        return {"promoted": self.promoted, "failed_gates": list(self.failed_gates),
                "notes": list(self.notes)}


def evaluate_promotion(*, model_metric: float, baseline_metric: float, metric_name: str,
                       latency_ms: float, latency_budget_ms: float, memory_bytes: float,
                       memory_budget_bytes: float, interval_coverage: float | None = None,
                       coverage_target: float = 0.9, safety_tests_passed: bool = True,
                       complements_baseline: bool = False) -> PromotionDecision:
    """Evaluate the promotion gates; lower error alone is never sufficient."""
    failed: list[str] = []
    notes: list[str] = []
    beats = model_metric < baseline_metric
    if not (beats or complements_baseline):
        failed.append("baseline_comparison")
        notes.append(f"{metric_name}: model {model_metric:.6g} vs baseline "
                     f"{baseline_metric:.6g}")
    if latency_ms > latency_budget_ms:
        failed.append("latency_budget")
    if memory_bytes > memory_budget_bytes:
        failed.append("memory_budget")
    if interval_coverage is not None and interval_coverage + 0.05 < coverage_target:
        failed.append("calibration")
        notes.append(f"interval coverage {interval_coverage:.3f} below target "
                     f"{coverage_target:.3f}")
    if not safety_tests_passed:
        failed.append("safety_tests")
    return PromotionDecision(promoted=not failed, failed_gates=tuple(failed),
                             notes=tuple(notes))


class TemporalModelRegistry:
    """An in-memory model-artifact registry with lifecycle transitions (Step 50)."""

    def __init__(self) -> None:
        """Initialize an empty registry."""
        self._records: dict[str, ModelArtifactRecord] = {}

    def register(self, record: ModelArtifactRecord, *, replace: bool = False) -> None:
        """Register an artifact; immutable artifacts cannot be silently replaced."""
        if record.artifact_hash in self._records and not replace:
            raise ValueError(f"artifact {record.artifact_hash} is already registered")
        self._records[record.artifact_hash] = record

    def get(self, artifact_hash: str) -> ModelArtifactRecord | None:
        """Return a registered artifact record."""
        return self._records.get(artifact_hash)

    def list_records(self) -> tuple[ModelArtifactRecord, ...]:
        """List all registered artifact records."""
        return tuple(self._records.values())

    def transition(self, artifact_hash: str, new_state: LifecycleState) -> ModelArtifactRecord:
        """Apply a lifecycle transition and return the updated record."""
        record = self._records.get(artifact_hash)
        if record is None:
            raise KeyError(f"artifact {artifact_hash} is not registered")
        allowed = {
            LifecycleState.PROPOSED: (LifecycleState.VALIDATED, LifecycleState.DEPRECATED),
            LifecycleState.VALIDATED: (LifecycleState.APPROVED, LifecycleState.DEPRECATED),
            LifecycleState.APPROVED: (LifecycleState.DEPRECATED,),
            LifecycleState.DEPRECATED: (),
        }
        if new_state not in allowed[record.approval_state]:
            raise ValueError(f"illegal lifecycle transition "
                             f"{record.approval_state.value} -> {new_state.value}")
        updated = ModelArtifactRecord(
            artifact_hash=record.artifact_hash,
            family=record.family,
            architecture=record.architecture,
            dataset_lineage=record.dataset_lineage,
            training_config=record.training_config,
            validation_metrics=record.validation_metrics,
            license=record.license,
            limitations=record.limitations,
            approval_state=new_state,
            trained_artifact_immutable=record.trained_artifact_immutable,
        )
        self._records[artifact_hash] = updated
        return updated

    def summary(self) -> dict[str, Any]:
        """Summarize the registry state."""
        counts: dict[str, int] = {state.value: 0 for state in LifecycleState}
        for record in self._records.values():
            counts[record.approval_state.value] += 1
        return {"n_records": len(self._records), "states": counts}
