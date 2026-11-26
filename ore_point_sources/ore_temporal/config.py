"""Versioned configuration for the Point 13 temporal and event-driven intelligence layer.

Every solver tolerance, threshold, time constant, and budget lives here rather than in
hidden constants, as required by the Point 13 parameter-governance section.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
import hashlib
import json
from typing import Any, Mapping

__all__ = ("TemporalConfig", "config_hash", "SOFTWARE_VERSIONS")

SOFTWARE_VERSIONS: dict[str, str] = {
    "ore_temporal": "1.0.0",
    "specification": "Point 13 TEDIL (canonical relevance rank 23 of 24)",
}


def _canonical_json(value: Any) -> str:
    """Serialize deterministically for hashing."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def config_hash(config: "TemporalConfig") -> str:
    """Return a stable SHA-256 identity for a configuration."""
    return hashlib.sha256(_canonical_json(config.to_dict()).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class TemporalConfig:
    """Governed parameters for ingestion, solvers, models, events, and governance phases."""

    # Phase A -- admission and selection
    seed: int = 0
    enable_neural_families: bool = True
    baseline_only: bool = False
    enable_feature_flag_rollback: bool = True
    enable_requested_family_override: bool = False

    # Phase B -- ingestion
    reorder_buffer_size: int = 16
    reorder_window_seconds: float = 5.0
    maximum_gap_seconds: float = 60.0
    event_count_baseline: int = 1000
    max_ingest_observations: int = 4096
    backpressure_policy: str = "aggregate"
    normalization_version: str = "znorm-v1"
    normalization_epsilon: float = 1e-8
    interpolation_max_gap: int = 3

    # Phase C -- model execution
    state_size_limit_bytes: int = 1_048_576
    state_retention_seconds: float = 3600.0
    state_checkpoint_enabled: bool = True
    max_sequence_length: int = 10_000
    max_solver_evaluations: int = 10_000
    solver_method: str = "euler"
    solver_max_step: float = 0.1
    solver_tolerance: float = 1e-6
    state_norm_bound: float = 1e6
    derivative_bound: float = 1e4
    max_events_per_run: int = 10_000
    max_spikes_per_run: int = 100_000
    inference_timeout_seconds: float = 30.0
    max_retries: int = 1

    # Model-family construction parameters
    lnn_time_constant: float = 1.0
    cfc_time_constant: float = 1.0
    ode_hidden_units: int = 16
    ssm_state_dim: int = 8
    snn_membrane_tau: float = 0.02
    snn_threshold: float = 1.0
    snn_reset_voltage: float = 0.0
    snn_rest_voltage: float = 0.0
    snn_membrane_resistance: float = 1.0
    snn_refractory_seconds: float = 0.002
    snn_encoding: str = "rate"
    snn_rate_window: float = 0.1

    # Phase D -- interpretation and event detection
    change_residual_threshold: float = 2.5
    cusum_drift: float = 0.5
    cusum_threshold: float = 5.0
    cusum_warmup: int = 8
    anomaly_z_threshold: float = 3.5
    expected_false_alarm_rate: float = 0.05
    coverage_target: float = 0.9
    max_actionable_events_per_run: int = 8
    high_impact_requires_deterministic_evidence: bool = True
    lag_correlation_max_lag: int = 10

    # Phase E -- cognitive integration
    publish_to_workspace: bool = True
    workspace_field: str = "artifact_references"
    notify_coordinator: bool = True
    max_coordinator_message_events: int = 8

    # Phase F -- portability
    enable_portable_graph_validation: bool = True
    enable_lava_adapter: bool = False
    portable_graph_tolerance: float = 1e-9

    # Phase G -- governance
    enable_controlled_adaptation: bool = False
    adaptation_learning_rate_bound: float = 0.01
    drift_review_threshold: float = 0.25
    drift_window: int = 64
    calibrate_uncertainty: bool = True
    calibration_version: str = "conformal-v1"
    raw_payload_logging: bool = False

    def __post_init__(self) -> None:
        """Validate governed parameters against their documented allowed ranges."""
        positive_ints = (
            "reorder_buffer_size", "event_count_baseline", "max_ingest_observations",
            "max_solver_evaluations", "max_sequence_length", "max_events_per_run",
            "max_spikes_per_run", "ode_hidden_units", "ssm_state_dim",
            "cusum_warmup", "max_actionable_events_per_run", "max_coordinator_message_events",
            "interpolation_max_gap", "lag_correlation_max_lag", "drift_window",
            "state_size_limit_bytes", "max_retries",
        )
        for name in positive_ints:
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        positive_floats = (
            "reorder_window_seconds", "maximum_gap_seconds", "solver_max_step", "state_norm_bound",
            "derivative_bound", "inference_timeout_seconds", "lnn_time_constant",
            "cfc_time_constant", "snn_membrane_tau", "snn_threshold", "snn_rate_window",
            "snn_refractory_seconds", "change_residual_threshold", "cusum_drift",
            "cusum_threshold", "anomaly_z_threshold", "lag_correlation_max_lag",
            "state_retention_seconds", "solver_tolerance", "normalization_epsilon",
        )
        for name in positive_floats:
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if self.solver_method not in ("euler", "rk4"):
            raise ValueError("solver_method must be 'euler' or 'rk4'")
        if self.backpressure_policy not in ("aggregate", "sample", "reject_late", "block"):
            raise ValueError("backpressure_policy must be a documented BackpressurePolicy value")
        if self.snn_encoding not in ("rate", "ttfs"):
            raise ValueError("snn_encoding must be 'rate' or 'ttfs'")
        if not 0.0 < self.coverage_target < 1.0:
            raise ValueError("coverage_target must be in (0, 1)")
        if not 0.0 <= self.expected_false_alarm_rate <= 1.0:
            raise ValueError("expected_false_alarm_rate must be in [0, 1]")
        if not 0.0 < self.adaptation_learning_rate_bound <= 0.1:
            raise ValueError("adaptation_learning_rate_bound must be in (0, 0.1]")

    def solver_settings(self) -> dict[str, Any]:
        """Return the versioned solver settings recorded in execution records (Step 7)."""
        return {
            "method": self.solver_method,
            "max_step": self.solver_max_step,
            "tolerance": self.solver_tolerance,
            "max_evaluations": self.max_solver_evaluations,
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize the configuration deterministically."""
        return {key: (value.value if hasattr(value, "value") else value)
                for key, value in asdict(self).items()}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TemporalConfig":
        """Restore a configuration from a serialized mapping."""
        allowed = {f for f in cls.__dataclass_fields__}  # noqa: C416 -- readable field filter
        return cls(**{key: item for key, item in value.items() if key in allowed})


def _is_dataclass_value(value: Any) -> bool:
    """Report whether the value is a dataclass instance."""
    return is_dataclass(value) and not isinstance(value, type)
