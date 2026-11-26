"""Edge deployment decisions and measured compression (Section 5, Steps 38-40)."""
from .compression import (  # noqa: F401
    CompressionPlan,
    CompressionReport,
    compress_for_edge,
    prune_weights,
    pruning_sparsity,
)
from .deployment import (  # noqa: F401
    EdgeDecision,
    EdgeDeploymentPlanner,
    SplitExecutionPlan,
)

__all__ = (
    "CompressionPlan",
    "CompressionReport",
    "EdgeDecision",
    "EdgeDeploymentPlanner",
    "SplitExecutionPlan",
    "compress_for_edge",
    "prune_weights",
    "pruning_sparsity",
)
