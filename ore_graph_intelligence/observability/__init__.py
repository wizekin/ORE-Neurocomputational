"""Observability: lifecycle events, metrics, and candidate lifecycle audit (step 40)."""
from .audit import AuditTrail, CandidateAudit, CandidateLifecycleAudit, LIFECYCLE_STAGES
from .events import EVENT_KINDS, GraphEventRecorder, GraphLifecycleEvent
from .metrics import (
    MetricSnapshot,
    SemanticSafetyMetrics,
    expected_calibration_error,
    false_positive_rate,
    graph_prediction_metrics,
    hits_at_k,
    mean_reciprocal_rank,
    precision_at_k,
    recall_at_k,
    relation_type_subgroup_metrics,
)

__all__ = [
    "AuditTrail", "CandidateAudit", "CandidateLifecycleAudit", "EVENT_KINDS",
    "GraphEventRecorder", "GraphLifecycleEvent", "LIFECYCLE_STAGES",
    "MetricSnapshot", "SemanticSafetyMetrics", "expected_calibration_error",
    "false_positive_rate", "graph_prediction_metrics", "hits_at_k",
    "mean_reciprocal_rank", "precision_at_k", "recall_at_k",
    "relation_type_subgroup_metrics",
]
