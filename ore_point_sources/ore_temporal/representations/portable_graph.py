"""NIR-inspired portable, hardware-independent model graphs (specification Steps 40-41).

A PortableGraph represents supported operations, state nodes, event channels, parameter
tensors, and metadata in a versioned structure that is independent of any backend.
Unsupported operators are rejected explicitly, and model semantics are kept separate
from device scheduling.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping

import numpy as np

__all__ = (
    "GraphNode",
    "PortableGraph",
    "SUPPORTED_GRAPH_OPERATIONS",
    "UnsupportedGraphOperationError",
)


SUPPORTED_GRAPH_OPERATIONS: frozenset[str] = frozenset({
    "input",
    "output",
    "linear",
    "tanh",
    "sigmoid",
    "exp_relax",          # closed-form exponential relaxation toward a target
    "ssm_cell",           # one selective state-space update step
    "lif_cell",           # one leaky integrate-and-fire advance step
    "euler_step",         # one explicit Euler integration step
    "rk4_step",           # one Runge-Kutta-4 integration step
    "threshold",          # deterministic threshold comparison
    "gate",               # elementwise input-dependent gating
})


class UnsupportedGraphOperationError(ValueError):
    """Raised when a graph references an operation the contract does not support."""


@dataclass(frozen=True, slots=True)
class GraphNode:
    """One node in a portable graph: an operation with typed inputs and parameters."""

    node_id: str
    operation: str
    inputs: tuple[str, ...]
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the node against the supported operation contract."""
        if not self.node_id or not isinstance(self.node_id, str):
            raise ValueError("node_id must be a non-empty string")
        if self.operation not in SUPPORTED_GRAPH_OPERATIONS:
            raise UnsupportedGraphOperationError(
                f"operation {self.operation!r} is not part of the portable contract; "
                f"supported operations: {sorted(SUPPORTED_GRAPH_OPERATIONS)}")
        for name in self.inputs:
            if not isinstance(name, str) or not name:
                raise ValueError("node inputs must be non-empty strings")

    def to_dict(self) -> dict[str, Any]:
        """Serialize the node with parameter shapes instead of raw payloads when asked."""
        return {
            "node_id": self.node_id,
            "operation": self.operation,
            "inputs": list(self.inputs),
            "parameters": _serialize_parameters(self.parameters),
        }


def _serialize_parameters(parameters: Mapping[str, Any]) -> dict[str, Any]:
    """Serialize parameters, converting arrays to nested lists."""
    output: dict[str, Any] = {}
    for key, value in parameters.items():
        if isinstance(value, np.ndarray):
            output[key] = value.tolist()
        elif isinstance(value, Mapping):
            output[key] = _serialize_parameters(value)
        elif isinstance(value, float) and not math.isfinite(value):
            raise ValueError("graph parameters must be finite")
        else:
            output[key] = value
    return output


@dataclass(frozen=True, slots=True)
class PortableGraph:
    """A versioned, backend-independent model graph (the architectural lesson of NIR)."""

    graph_id: str
    graph_version: str
    nodes: tuple[GraphNode, ...]
    state_nodes: tuple[str, ...]
    event_channels: tuple[str, ...]
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate node identity, reachability bookkeeping, and operation support."""
        if not self.graph_id or not isinstance(self.graph_id, str):
            raise ValueError("graph_id must be a non-empty string")
        node_ids = [node.node_id for node in self.nodes]
        if len(set(node_ids)) != len(node_ids):
            raise ValueError("graph node ids must be unique")
        known = set(node_ids) | set(self.input_names) | {"__dt__"}
        for node in self.nodes:
            for source in node.inputs:
                if source not in known:
                    raise ValueError(f"node {node.node_id} references unknown input {source!r}")
        for name in self.output_names:
            if name not in node_ids and name not in self.input_names:
                raise ValueError(f"output {name!r} is not produced by any node")

    def validate_supported_operations(self) -> None:
        """Explicitly reject unsupported operators (Step 41)."""
        for node in self.nodes:
            if node.operation not in SUPPORTED_GRAPH_OPERATIONS:
                raise UnsupportedGraphOperationError(node.operation)

    def parameter_tensor_shapes(self) -> dict[str, list[int]]:
        """Report parameter tensor shapes without exporting payloads."""
        shapes: dict[str, list[int]] = {}
        for node in self.nodes:
            for key, value in node.parameters.items():
                if isinstance(value, np.ndarray):
                    shapes[f"{node.node_id}.{key}"] = list(value.shape)
        return shapes

    def to_dict(self) -> dict[str, Any]:
        """Serialize the full graph deterministically."""
        return {
            "graph_id": self.graph_id,
            "graph_version": self.graph_version,
            "nodes": [node.to_dict() for node in self.nodes],
            "state_nodes": list(self.state_nodes),
            "event_channels": list(self.event_channels),
            "input_names": list(self.input_names),
            "output_names": list(self.output_names),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PortableGraph":
        """Restore a graph from its serialized form."""
        nodes = tuple(
            GraphNode(
                node_id=item["node_id"],
                operation=item["operation"],
                inputs=tuple(item["inputs"]),
                parameters=_restore_parameters(item.get("parameters", {})),
            )
            for item in value["nodes"]
        )
        return cls(
            graph_id=value["graph_id"],
            graph_version=value["graph_version"],
            nodes=nodes,
            state_nodes=tuple(value.get("state_nodes", ())),
            event_channels=tuple(value.get("event_channels", ())),
            input_names=tuple(value.get("input_names", ())),
            output_names=tuple(value.get("output_names", ())),
            metadata=dict(value.get("metadata", {})),
        )


def _restore_parameters(parameters: Mapping[str, Any]) -> dict[str, Any]:
    """Restore numpy parameter tensors from nested lists."""
    output: dict[str, Any] = {}
    for key, value in parameters.items():
        if isinstance(value, list) and value and isinstance(value[0], list):
            output[key] = np.asarray(value, dtype=float)
        else:
            output[key] = value
    return output
