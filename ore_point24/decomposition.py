"""Phase IV, Step 17 - decompose the task into verifiable subtasks.

Work backward from each required output artifact to its exact input
contracts, decompose only where a subtask has a clear responsibility or an
independently verifiable result, add explicit verification and integration
subtasks, and keep human approval steps as first-class nodes.  The result is
a directed graph ``G = (V, E)`` in which every node is a task or module
execution and every edge is a typed artifact dependency.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from .core import new_id, sha256_digest
from .task_requests import TaskRequest

DECOMPOSITION_SCHEMA = "ore.point24.decomposition.v1"

VERIFICATION_NODE = "verification"
APPROVAL_NODE = "approval"
MODULE_NODE = "module"


@dataclass(frozen=True, slots=True)
class SubtaskNode:
    """One node of the Step 17 task graph."""

    node_id: str
    node_type: str  # module | verification | approval
    responsibility: str
    required_capability: str | None
    produces_contract: str | None = None
    consumes_contracts: tuple[str, ...] = ()
    provider_hint: str | None = None
    reusable: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "node_type": self.node_type,
            "responsibility": self.responsibility,
            "required_capability": self.required_capability,
            "produces_contract": self.produces_contract,
            "consumes_contracts": list(self.consumes_contracts),
            "provider_hint": self.provider_hint,
            "reusable": self.reusable,
        }


@dataclass(slots=True)
class TaskGraph:
    """The directed graph G = (V, E) produced by decomposition."""

    task_request_id: str
    nodes: dict[str, SubtaskNode] = field(default_factory=dict)
    edges: dict[str, tuple[str, ...]] = field(default_factory=dict)  # consumer -> producers

    def add_node(self, node: SubtaskNode) -> str:
        if node.node_id in self.nodes:
            raise ValueError(f"duplicate node: {node.node_id}")
        self.nodes[node.node_id] = node
        self.edges.setdefault(node.node_id, ())
        return node.node_id

    def add_edge(self, consumer: str, producer: str) -> None:
        if consumer not in self.nodes or producer not in self.nodes:
            raise KeyError("both endpoints must exist")
        current = tuple(self.edges.get(consumer, ()))
        if producer not in current:
            self.edges[consumer] = current + (producer,)

    def producers_of(self, node_id: str) -> tuple[str, ...]:
        return tuple(self.edges.get(node_id, ()))

    def shared_dependencies(self) -> dict[str, list[str]]:
        """Nodes producing artifacts consumed by more than one consumer."""
        consumers: dict[str, list[str]] = {}
        for consumer, producers in self.edges.items():
            for producer in producers:
                consumers.setdefault(producer, []).append(consumer)
        return {producer: sorted(users) for producer, users in consumers.items() if len(users) > 1}

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": DECOMPOSITION_SCHEMA,
            "task_request_id": self.task_request_id,
            "nodes": {key: node.to_dict() for key, node in sorted(self.nodes.items())},
            "edges": {key: list(value) for key, value in sorted(self.edges.items())},
            "shared_dependencies": self.shared_dependencies(),
        }

    def digest(self) -> str:
        return sha256_digest(self.to_dict())


class TaskDecomposer:
    """Backward-chaining decomposition from required outputs to input contracts."""

    def __init__(self) -> None:
        self._responsibilities: dict[str, tuple[str, str, str]] = {}
        self._consumes: dict[str, tuple[str, ...]] = {}

    def register_capability_route(
        self,
        capability: str,
        *,
        responsibility: str,
        produces_contract: str,
        consumes_contracts: tuple[str, ...] = (),
        provider_hint: str | None = None,
    ) -> None:
        """Record how a declared capability is realised by the architecture."""
        self._responsibilities[capability] = (responsibility, produces_contract, provider_hint or "")
        self._consumes[capability] = tuple(consumes_contracts)

    def decompose(self, request: TaskRequest, *, add_verification: bool = True) -> TaskGraph:
        """Decompose a typed request into a graph with verification nodes.

        Decomposition never duplicates an existing module's responsibility
        without a documented reason, and every capability requirement becomes
        exactly one module node (shared dependencies are recorded once).
        """
        graph = TaskGraph(task_request_id=request.task_request_id)
        output_node_id: list[str] = []
        for index, capability in enumerate(request.required_capabilities):
            responsibility, produces, hint = self._responsibilities.get(
                capability, (f"Fulfil capability {capability}", None, "")
            )
            consumes = self._consumes.get(capability, ())
            node_id = f"step_{index + 1:02d}_{capability.replace('.', '_').replace('-', '_')}"
            graph.add_node(SubtaskNode(
                node_id=node_id,
                node_type=MODULE_NODE,
                responsibility=responsibility,
                required_capability=capability,
                produces_contract=produces,
                consumes_contracts=consumes,
                provider_hint=hint or None,
                reusable=index > 0 and capability in {c for c in request.optional_capabilities},
            ))
            output_node_id.append(node_id)

        if add_verification:
            graph.add_node(SubtaskNode(
                node_id="verify_outputs",
                node_type=VERIFICATION_NODE,
                responsibility=(
                    "Independently verify schemas, provenance, logical validity and "
                    "task-specific acceptance conditions of every produced artifact"
                ),
                required_capability=None,
                produces_contract="P24-VERIFICATION-REPORT",
                consumes_contracts=tuple(
                    node.produces_contract for node in graph.nodes.values()
                    if node.produces_contract
                ),
            ))
            for producer in output_node_id:
                graph.add_edge("verify_outputs", producer)

        if request.permission_policy == "requires_human_approval":
            graph.add_node(SubtaskNode(
                node_id="human_approval",
                node_type=APPROVAL_NODE,
                responsibility="Explicit human approval checkpoint before consequential execution",
                required_capability=None,
                produces_contract=None,
            ))
            if "verify_outputs" in graph.nodes:
                graph.add_edge("human_approval", "verify_outputs")

        return graph


__all__ = [
    "DECOMPOSITION_SCHEMA",
    "SubtaskNode",
    "TaskDecomposer",
    "TaskGraph",
]
