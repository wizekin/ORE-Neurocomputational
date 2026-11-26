"""Versioned configuration for the Point 15 agentic workflow and memory layer.

Every delegation bound, retry constant, risk threshold, protocol limit, memory
retention class, and governance budget lives here rather than in hidden
constants, as required by the Point 15 parameter-governance sections.  The
configuration is deliberately small enough to audit by reading one file.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
import hashlib
import json
from typing import Any, Mapping

__all__ = (
    "AgenticConfig",
    "RetrievalWeights",
    "RiskWeights",
    "WorkerScoreWeights",
    "config_hash",
    "default_config",
    "SOFTWARE_VERSIONS",
)

SOFTWARE_VERSIONS: dict[str, str] = {
    "ore_agentic": "1.0.0",
    "specification": (
        "Point 15 Agentic Systems, Cognitive Memory Architectures and "
        "Communication Protocols (canonical relevance rank 4 of 24)"
    ),
    "workflow": "1.0.0",
    "delegation": "1.0.0",
    "registry": "1.0.0",
    "memory": "1.0.0",
    "retrieval": "1.0.0",
    "protocols": "1.0.0",
    "evaluation": "1.0.0",
    "observability": "1.0.0",
    "security": "1.0.0",
    "persistence": "1.0.0",
}


def _canonical_json(value: Any) -> str:
    """Serialize deterministically for hashing."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def config_hash(config: AgenticConfig) -> str:
    """Return a stable SHA-256 identity for a configuration."""
    return hashlib.sha256(_canonical_json(config.to_dict()).encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True, slots=True)
class RiskWeights:
    """Weights for the configurable risk score R = sum(w_i * r_i) (Step 3)."""

    sensitivity: float = 0.30
    irreversibility: float = 0.30
    uncertainty: float = 0.20
    external_impact: float = 0.20

    def __post_init__(self) -> None:
        """Require a convex combination of non-negative weights."""
        values = (self.sensitivity, self.irreversibility, self.uncertainty,
                  self.external_impact)
        if any(weight < 0.0 for weight in values):
            raise ValueError("risk weights must be non-negative")
        if abs(sum(values) - 1.0) > 1e-9:
            raise ValueError("risk weights must sum to 1.0")


@dataclass(frozen=True, slots=True)
class WorkerScoreWeights:
    """Weights for the worker-selection objective S(a, t) (Step 11).

    S = w_f*F + w_q*Q - w_l*L - w_c*C - w_u*U - w_r*R with fit F, measured
    quality Q, latency L, cost C, uncertainty U, and load R.
    """

    fit: float = 0.40
    quality: float = 0.30
    latency: float = 0.10
    cost: float = 0.10
    uncertainty: float = 0.05
    load: float = 0.05

    def __post_init__(self) -> None:
        """Require non-negative weights (the objective is not a probability)."""
        values = (self.fit, self.quality, self.latency, self.cost,
                  self.uncertainty, self.load)
        if any(weight < 0.0 for weight in values):
            raise ValueError("worker score weights must be non-negative")


@dataclass(frozen=True, slots=True)
class RetrievalWeights:
    """Weights for the hybrid retrieval score R(d, q) (Step 41).

    R = alpha*R_lex + beta*R_semantic + gamma*R_graph + delta*R_fresh
        + epsilon*R_authority.  Component scores are normalized to [0, 1]
    before weighting; the blend is a ranking score, not a probability.
    """

    lexical: float = 0.25
    semantic: float = 0.25
    graph: float = 0.20
    freshness: float = 0.15
    authority: float = 0.15

    def __post_init__(self) -> None:
        """Require a convex combination of non-negative weights."""
        values = (self.lexical, self.semantic, self.graph, self.freshness,
                  self.authority)
        if any(weight < 0.0 for weight in values):
            raise ValueError("retrieval weights must be non-negative")
        if abs(sum(values) - 1.0) > 1e-9:
            raise ValueError("retrieval weights must sum to 1.0")


@dataclass(frozen=True, slots=True)
class AgenticConfig:
    """Governed parameters for all eight phases of the 50-step algorithm.

    Phase I -- goal intake and workflow compilation; Phase II -- registry,
    delegation and parallel execution; Phase III -- state, recovery and
    side-effect safety; Phase IV -- evaluator-optimizer; Phase V -- MCP and
    A2A interoperability; Phase VI -- three-tier memory and RAG; Phase VII --
    production reliability; Phase VIII -- governance.
    """

    # Phase I -- intake and compilation
    seed: int = 0
    max_objective_chars: int = 4000
    max_tasks_per_workflow: int = 64
    default_deadline_seconds: float = 600.0
    risk_weights: RiskWeights = RiskWeights()
    automatic_risk_threshold: float = 0.35
    approval_risk_threshold: float = 0.70
    ambiguity_requires_clarification_when_high_risk: bool = True
    request_fingerprint_fields: tuple[str, ...] = ("objective", "inputs", "deliverables")

    # Phase II -- delegation and parallel execution
    worker_pool_size: int = 4
    per_workflow_concurrency: int = 4
    per_worker_concurrency: int = 2
    global_concurrency: int = 8
    queue_capacity: int = 32
    worker_score_weights: WorkerScoreWeights = WorkerScoreWeights()
    lease_ttl_seconds: float = 60.0
    heartbeat_interval_seconds: float = 5.0

    # Retry policy defaults (Step 16)
    retry_max_attempts: int = 3
    retry_initial_delay_seconds: float = 0.05
    retry_max_delay_seconds: float = 2.0
    retry_jitter_seconds: float = 0.02
    retry_total_time_budget_seconds: float = 8.0

    # Phase III -- state, checkpoints, side effects
    checkpoints_enabled: bool = True
    checkpoint_every_n_tasks: int = 4
    max_checkpoints_retained: int = 8
    reconciliation_required_before_retry: bool = True

    # Phase IV -- evaluator-optimizer
    critic_max_iterations: int = 3
    critic_cost_budget: float = 16.0
    critic_stall_threshold: float = 0.0

    # Phase V -- protocol interoperability
    mcp_enabled: bool = True
    mcp_read_write_scopes_enabled: bool = False
    mcp_max_response_bytes: int = 262_144
    protocol_default_deadline_seconds: float = 30.0
    a2a_enabled: bool = False

    # Phase VI -- memory and retrieval
    retrieval_weights: RetrievalWeights = RetrievalWeights()
    episode_max_payload_chars: int = 2_000
    episode_default_retention_class: str = "standard"
    procedure_min_passing_tests: int = 1
    semantic_proposals_start_as_candidate: bool = True

    # Phase VII -- reliability and governance
    tracing_enabled: bool = True
    semantic_cache_enabled: bool = True
    semantic_cache_ttl_seconds: float = 300.0
    drift_window: int = 32
    drift_review_threshold: float = 0.25
    drift_quarantine_allowed: bool = False
    quota_cpu_units: float = 32.0
    quota_memory_mb: float = 4096.0
    quota_seconds: float = 300.0
    quota_cost: float = 100.0
    sandbox_wall_clock_seconds: float = 10.0
    sandbox_max_output_bytes: int = 65_536
    max_events_per_run: int = 10_000

    # Phase VIII -- workspace integration
    publish_to_workspace: bool = True
    workspace_field: str = "artifact_references"

    def __post_init__(self) -> None:
        """Validate governed parameters against their documented ranges."""
        positive_ints = (
            "max_objective_chars", "max_tasks_per_workflow", "worker_pool_size",
            "per_workflow_concurrency", "per_worker_concurrency", "global_concurrency",
            "queue_capacity", "retry_max_attempts", "checkpoint_every_n_tasks",
            "max_checkpoints_retained", "critic_max_iterations", "mcp_max_response_bytes",
            "episode_max_payload_chars", "procedure_min_passing_tests", "drift_window",
            "max_events_per_run", "sandbox_max_output_bytes",
        )
        for name in positive_ints:
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        positive_floats = (
            "default_deadline_seconds", "lease_ttl_seconds", "heartbeat_interval_seconds",
            "retry_initial_delay_seconds", "retry_max_delay_seconds",
            "retry_total_time_budget_seconds", "critic_cost_budget",
            "protocol_default_deadline_seconds", "semantic_cache_ttl_seconds",
            "quota_cpu_units", "quota_memory_mb", "quota_seconds", "quota_cost",
            "sandbox_wall_clock_seconds",
        )
        for name in positive_floats:
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if float(self.retry_jitter_seconds) < 0.0:
            raise ValueError("retry_jitter_seconds must be non-negative")
        for name in ("automatic_risk_threshold", "approval_risk_threshold",
                     "drift_review_threshold"):
            if not 0.0 <= float(getattr(self, name)) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.automatic_risk_threshold > self.approval_risk_threshold:
            raise ValueError(
                "automatic_risk_threshold must not exceed approval_risk_threshold")
        if self.retry_initial_delay_seconds > self.retry_max_delay_seconds:
            raise ValueError(
                "retry_initial_delay_seconds must not exceed retry_max_delay_seconds")
        if self.workspace_field not in ("artifact_references", "working_memory",
                                        "module_status", "unresolved_questions"):
            raise ValueError("workspace_field must be an append-only workspace field")

    def retry_policy_dict(self) -> dict[str, Any]:
        """Return the versioned retry settings recorded in task records (Step 16)."""
        return {
            "max_attempts": self.retry_max_attempts,
            "initial_delay_seconds": self.retry_initial_delay_seconds,
            "max_delay_seconds": self.retry_max_delay_seconds,
            "jitter_seconds": self.retry_jitter_seconds,
            "total_time_budget_seconds": self.retry_total_time_budget_seconds,
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize the configuration deterministically."""
        def _expand(value: Any) -> Any:
            if is_dataclass(value) and not isinstance(value, type):
                return {key: _expand(item) for key, item in asdict(value).items()}
            return value

        return {key: _expand(value) for key, value in asdict(self).items()}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> AgenticConfig:
        """Restore a configuration from a serialized mapping."""
        nested = {"risk_weights": RiskWeights,
                  "worker_score_weights": WorkerScoreWeights,
                  "retrieval_weights": RetrievalWeights}
        allowed = set(cls.__dataclass_fields__)
        kwargs: dict[str, Any] = {}
        for key, item in value.items():
            if key not in allowed:
                continue
            builder = nested.get(key)
            if builder is not None and isinstance(item, Mapping):
                kwargs[key] = builder(**item)
            else:
                kwargs[key] = item
        return cls(**kwargs)


def default_config() -> AgenticConfig:
    """Return the auditable default configuration."""
    return AgenticConfig()
