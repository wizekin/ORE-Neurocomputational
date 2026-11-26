"""Governance parameters for the Point 14 Replaceable Learned-Model Subsystem (RLMS).

Every tunable that controls model admission, routing, adaptation, generation,
explanation, federated learning, optimization, and lifecycle behaviour lives here so
that deployments can review, version, and hash one configuration object instead of
hunting for scattered constants.  Mirrors the ``ore_temporal.config`` convention.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from typing import Any, Mapping

__all__ = ("SOFTWARE_VERSIONS", "LearningConfig", "config_hash", "default_config")

SOFTWARE_VERSIONS: dict[str, str] = {
    "ore_learning": "1.0.0",
    "contracts": "1.0.0",
    "registry": "1.0.0",
    "routing": "1.0.0",
    "adapters": "1.0.0",
    "inference": "1.0.0",
    "adaptation": "1.0.0",
    "generation": "1.0.0",
    "explainability": "1.0.0",
    "rewards": "1.0.0",
    "federated": "1.0.0",
    "optimization": "1.0.0",
    "monitoring": "1.0.0",
    "audit": "1.0.0",
    "edge": "1.0.0",
    "pipeline": "1.0.0",
}


def _hashable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def config_hash(config: "LearningConfig | None") -> str:
    """Return a stable digest of the effective configuration (governance fingerprint)."""
    payload = _hashable_json(LearningConfig.default().to_dict() if config is None
                             else config.to_dict())
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True, slots=True)
class LearningConfig:
    """Versioned governance parameters for the RLMS pipeline (specification Section 5).

    The defaults encode the specification's guard rails: utility scores can never
    override hard eligibility, generated outputs are untrusted, learned explanations
    are never proof, rewards are training signals only, and high-risk promotions
    always require recorded human approval.
    """

    #: Step 8 -- utility weights, normalized inside the ranking module.
    utility_weight_quality: float = 0.30
    utility_weight_calibration: float = 0.15
    utility_weight_robustness: float = 0.15
    utility_weight_latency_penalty: float = 0.15
    utility_weight_resource_penalty: float = 0.10
    utility_weight_uncertainty_penalty: float = 0.15

    #: Steps 8/10 -- hard bounds that ranking and ensembles may not exceed.
    max_ensemble_size: int = 3
    #: Candidates within this utility margin of the top may form a bounded ensemble.
    min_utility_margin_for_ensemble: float = 0.05

    #: Step 9 -- MoE routing concentration alert threshold (max share of one expert).
    moe_max_expert_share: float = 0.80

    #: Step 11 -- safety margin applied to declared model resource profiles.
    resource_safety_margin: float = 0.20

    #: Step 15 -- isolated adaptation defaults.
    adaptation_seed: int = 20241123
    adaptation_epochs: int = 60
    adaptation_batch_size: int = 8
    adaptation_learning_rate: float = 0.05
    #: Step 16 -- minimum held-out improvement required for automatic promotion.
    min_adaptation_gain: float = 0.02
    #: Step 16 -- maximum accepted quality drop when a trade-off is approved.
    max_approved_tradeoff_quality_drop: float = 0.05

    #: Step 17 -- test-time training limits; never persisted into canonical artifacts.
    ttt_max_steps: int = 5
    ttt_max_seconds: float = 1.0
    ttt_max_samples: int = 16

    #: Step 19 -- reject modality payloads outside these declared limits.
    max_image_side_pixels: int = 4096
    max_audio_seconds: float = 3600.0
    max_video_frames: int = 10000

    #: Steps 23-26 -- generative data laboratory guard rails.
    gan_training_steps: int = 240
    gan_memorization_epsilon: float = 1e-3
    synthetic_min_diversity: float = 0.10
    synthetic_max_psi: float = 0.25
    sandbox_allowed_modules: tuple[str, ...] = ("math", "statistics", "json", "random",)

    #: Step 28-31 -- explanation approximation settings.
    shap_max_exact_features: int = 10
    shap_sampling_background_size: int = 32
    lime_num_samples: int = 64
    lime_kernel_width: float = 0.75
    counterfactual_max_features_changed: int = 3

    #: Steps 32-37 -- process validation and verifiable rewards.
    prm_screening_threshold: float = 0.5
    rlvr_step_weight: float = 1.0
    rlvr_final_weight: float = 4.0
    rlvr_violation_weight: float = 8.0
    hidden_test_fraction: float = 0.4

    #: Steps 41-43 -- federated learning controls.
    federated_min_cohort: int = 2
    federated_clip_norm: float = 1.0
    federated_dp_noise_multiplier: float = 0.01
    federated_require_secure_aggregation: bool = True

    #: Steps 44-47 -- QUBO / Ising guard rails.
    qubo_penalty_rho: float = 10.0
    qubo_exhaustive_verify_max_variables: int = 16
    annealing_initial_temperature: float = 2.0
    annealing_steps: int = 400

    #: Steps 48-52 -- monitoring and lifecycle thresholds.
    calibration_ece_alert: float = 0.10
    drift_psi_alert: float = 0.25
    drift_min_sample_size: int = 30
    drift_canary_quality_alert: float = 0.10
    behavior_max_schema_violation_rate: float = 0.02
    canary_default_fraction: float = 0.2
    canary_regression_rollback_threshold: float = 0.05

    #: Steps 55-56 -- approval and retirement policy.
    high_risk_requires_approval: bool = True
    retirement_preserves_lineage: bool = True

    #: Step 38-40 -- edge deployment thresholds.
    edge_retention_threshold: float = 0.95
    edge_max_memory_mb: float = 8.0

    #: Workspace publication field for the shared cognitive workspace bridge.
    workspace_field: str = "artifact_references"

    #: Where the audit SQLite database lives when no path is supplied.
    audit_database_filename: str = "ore_learning_lifecycle.sqlite3"

    def __post_init__(self) -> None:
        """Validate the governance parameters once at construction time."""
        if not 0 < self.max_ensemble_size <= 8:
            raise ValueError("max_ensemble_size must be in [1, 8]")
        if not 0.0 <= self.resource_safety_margin <= 1.0:
            raise ValueError("resource_safety_margin must be a fraction in [0, 1]")
        if not 0.0 <= self.moe_max_expert_share <= 1.0:
            raise ValueError("moe_max_expert_share must be a fraction in [0, 1]")
        for name in ("utility_weight_quality", "utility_weight_calibration",
                     "utility_weight_robustness", "utility_weight_latency_penalty",
                     "utility_weight_resource_penalty", "utility_weight_uncertainty_penalty"):
            value = getattr(self, name)
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if self.ttt_max_steps <= 0 or self.ttt_max_samples <= 0:
            raise ValueError("test-time-training limits must be positive")
        if self.federated_min_cohort < 1:
            raise ValueError("federated_min_cohort must be at least 1")
        if not 0.0 < self.canary_default_fraction < 1.0:
            raise ValueError("canary_default_fraction must be in (0, 1)")
        if self.shap_max_exact_features < 2:
            raise ValueError("shap_max_exact_features must be at least 2")

    @classmethod
    def default(cls) -> "LearningConfig":
        """Return the canonical default configuration."""
        return cls()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dictionary view of the configuration."""
        return asdict(self)

    def describe(self) -> str:
        """Human-readable governance summary."""
        return (
            "RLMS governance: utility weights (q/c/r/l/k/u) = "
            f"({self.utility_weight_quality}, {self.utility_weight_calibration}, "
            f"{self.utility_weight_robustness}, {self.utility_weight_latency_penalty}, "
            f"{self.utility_weight_resource_penalty}, {self.utility_weight_uncertainty_penalty}); "
            f"ensemble <= {self.max_ensemble_size}; TTT capped at "
            f"{self.ttt_max_steps} steps; federated cohort >= {self.federated_min_cohort}; "
            f"canary fraction {self.canary_default_fraction}; high-risk promotions "
            f"require approval={self.high_risk_requires_approval}"
        )


def default_config() -> LearningConfig:
    """Return the canonical default configuration (specification defaults)."""
    return LearningConfig()
