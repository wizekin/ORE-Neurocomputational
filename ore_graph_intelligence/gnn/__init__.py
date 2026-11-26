"""Relational GNN stack: R-GCN first, with GCN/GraphSAGE/GAT as measured alternatives."""
from .gcn import GCNEncoder, GCNLayer, normalized_adjacency
from .gat import GATEncoder, GATLayer
from .graphsage import GraphSAGEEncoder, GraphSAGELayer
from .inference import (
    DeterministicGraphBaselines,
    GNNInferenceEngine,
    NODE_KIND_ORDER,
    OverSmoothingMonitor,
    OverSmoothingReport,
    RGCNLinkPredictor,
    RGCNTrainingReport,
    build_snapshot_features,
)
from .rgcn import DistMultDecoder, RGCNEncoder, RGCNLayer, RelationGraph

__all__ = [
    "DistMultDecoder", "GCNEncoder", "GCNLayer", "GATEncoder", "GATLayer",
    "GraphSAGEEncoder", "GraphSAGELayer", "RGCNEncoder", "RGCNLayer", "RelationGraph",
    "DeterministicGraphBaselines", "GNNInferenceEngine", "NODE_KIND_ORDER",
    "OverSmoothingMonitor", "OverSmoothingReport", "RGCNLinkPredictor",
    "RGCNTrainingReport", "build_snapshot_features", "normalized_adjacency",
]
