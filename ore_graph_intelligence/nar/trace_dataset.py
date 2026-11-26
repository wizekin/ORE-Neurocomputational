"""Trace datasets: deterministic inputs, intermediate states, outputs, versions, seeds (step 42-43).

A :class:`AlgorithmTrace` is the supervision unit for neural algorithmic reasoning: the
task input as tensors, the reference intermediate states (``step_states``), the final
targets, the supervision decision (stepwise versus final), and full provenance
(implementation version, parameters, seed). Traces are generated only from trusted
reference algorithms, never from learned models.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .task_generators import (
    NAR_REFERENCE_VERSION,
    NarrowTask,
    reference_solve,
)

#: Static node feature dimension (value, degree, is-source flag).
NODE_FEATURE_DIM = 3

#: Edge feature dimension per directed edge (weight, reciprocal indicator, uniform prior).
EDGE_FEATURE_DIM = 3

_DISTANCE_CAP = 10.0 * 20


def _edge_key(u: int, v: int) -> tuple[int, int]:
    """Canonical undirected edge key so both directions align with reference output."""
    return (u, v) if u <= v else (v, u)


def _normalize_weights(weights: np.ndarray) -> np.ndarray:
    return np.clip(weights / 10.0, 0.0, 1.0)


def _static_node_features(task: NarrowTask, *, degree: np.ndarray) -> np.ndarray:
    n_nodes = task.n_nodes
    if task.task_type == "sorting":
        values = np.array(task.sequence, dtype=np.float64)
        return np.stack([values, np.zeros(n_nodes), np.zeros(n_nodes)], axis=1)
    is_source = np.zeros(n_nodes, dtype=np.float64)
    is_source[task.source_node] = 1.0
    degree_norm = degree / max(1.0, float(degree.max()) if degree.size else 1.0)
    position = np.arange(n_nodes, dtype=np.float64) / max(1, n_nodes - 1)
    return np.stack([position, degree_norm, is_source], axis=1)


@dataclass(frozen=True, slots=True)
class AlgorithmTrace:
    """One supervised trace: task tensors plus reference states and targets."""

    task: NarrowTask
    node_features: np.ndarray
    edge_sources: np.ndarray
    edge_targets: np.ndarray
    edge_features: np.ndarray
    supervision: str
    target_kind: str
    initial_state: np.ndarray
    step_states: tuple[np.ndarray, ...]
    final_node_target: np.ndarray
    final_edge_target: np.ndarray
    implementation_version: str
    seed: int

    def __post_init__(self) -> None:
        if self.supervision not in {"stepwise", "final"}:
            raise ValueError("supervision must be stepwise or final")
        if self.target_kind not in {"node", "edge"}:
            raise ValueError("target_kind must be node or edge")
        n_nodes = self.task.n_nodes
        if self.node_features.shape != (n_nodes, NODE_FEATURE_DIM):
            raise ValueError("node_features must be (n_nodes, NODE_FEATURE_DIM)")
        if self.edge_sources.shape != self.edge_targets.shape:
            raise ValueError("edge source/target arrays must align")
        if self.edge_features.shape != (self.edge_sources.size, EDGE_FEATURE_DIM):
            raise ValueError("edge_features must be (n_edges, EDGE_FEATURE_DIM)")
        if self.initial_state.shape != (n_nodes,):
            raise ValueError("initial_state must be (n_nodes,)")

    # ------------------------------------------------------------------ helpers

    def n_edges(self) -> int:
        return int(self.edge_sources.size)

    def summary(self) -> dict[str, object]:
        """Return the manifest summary of this trace (step 24-style artifact metadata)."""
        return {
            "task_id": self.task.task_id,
            "task_type": self.task.task_type,
            "topology": self.task.topology,
            "n_nodes": self.task.n_nodes,
            "n_edges": self.n_edges(),
            "supervision": self.supervision,
            "target_kind": self.target_kind,
            "steps": len(self.step_states),
            "implementation_version": self.implementation_version,
            "seed": self.seed,
        }


def build_trace(task: NarrowTask) -> AlgorithmTrace:
    """Project one narrow task into tensors plus deterministic reference targets."""
    reference = reference_solve(task)
    if task.task_type == "sorting":
        values = np.array(task.sequence, dtype=np.float64)
        n = values.size
        sources = np.repeat(np.arange(n, dtype=np.int64), n)
        targets = np.tile(np.arange(n, dtype=np.int64), n)
        diff = values[sources] - values[targets]
        edge_features = np.stack([
            np.sign(diff),
            np.abs(diff),
            np.zeros(diff.size),
        ], axis=1)
        ranks = reference.node_output / max(1, n - 1)
        return AlgorithmTrace(
            task=task,
            node_features=np.stack([values, np.zeros(n), np.zeros(n)], axis=1),
            edge_sources=sources,
            edge_targets=targets,
            edge_features=edge_features,
            supervision="final",
            target_kind="node",
            initial_state=np.zeros(n, dtype=np.float64),
            step_states=(ranks.copy(),),
            final_node_target=ranks.copy(),
            final_edge_target=np.zeros(sources.size, dtype=np.float64),
            implementation_version=task.implementation_version,
            seed=task.seed,
        )

    sources, targets, weights = task.adjacency_undirected()
    degree = np.zeros(task.n_nodes, dtype=np.float64)
    np.add.at(degree, targets, 1.0)
    node_features = _static_node_features(task, degree=degree)
    edge_features = np.stack([
        _normalize_weights(weights),
        np.ones(weights.size),
        np.full(weights.size, 0.5),
    ], axis=1)

    if task.task_type == "minimum_spanning_tree":
        in_tree = np.zeros(sources.size, dtype=np.float64)
        mst_reference = reference.edge_output
        reference_by_key = {
            _edge_key(u, v): mst_reference[undirected_index]
            for undirected_index, (u, v, _w) in enumerate(task.edges)
        }
        for directed_index in range(sources.size):
            in_tree[directed_index] = reference_by_key.get(
                _edge_key(int(sources[directed_index]), int(targets[directed_index])), 0.0,
            )
        return AlgorithmTrace(
            task=task,
            node_features=node_features,
            edge_sources=sources,
            edge_targets=targets,
            edge_features=edge_features,
            supervision="final",
            target_kind="edge",
            initial_state=np.zeros(task.n_nodes, dtype=np.float64),
            step_states=(in_tree.copy(),),
            final_node_target=np.zeros(task.n_nodes, dtype=np.float64),
            final_edge_target=in_tree.copy(),
            implementation_version=task.implementation_version,
            seed=task.seed,
        )

    if task.task_type == "shortest_path":
        cap = float(task.n_nodes * 10.0) + 1.0
        states = tuple(np.clip(state, 0.0, cap) / cap for state in reference.step_states)
    elif task.task_type == "connectivity":
        states = tuple(state / max(1, task.n_nodes - 1) for state in reference.step_states)
    else:
        states = tuple(state.copy() for state in reference.step_states)

    final = states[-1]
    return AlgorithmTrace(
        task=task,
        node_features=node_features,
        edge_sources=sources,
        edge_targets=targets,
        edge_features=edge_features,
        supervision="stepwise",
        target_kind="node",
        initial_state=states[0].copy(),
        step_states=states,
        final_node_target=final.copy(),
        final_edge_target=np.zeros(sources.size, dtype=np.float64),
        implementation_version=task.implementation_version,
        seed=task.seed,
    )


class TraceDataset:
    """Bounded collection of traces with deterministic ordering and split bookkeeping.

    Splits follow the NAR leakage policy: hold out *sizes and topologies*, not random
    traces, so that interpolation (seen sizes/topologies) is always separable from size
    extrapolation and unseen-topology generalization (step 45).
    """

    def __init__(self, traces: Sequence[AlgorithmTrace], *, policy: str = "size_topology") -> None:
        self.traces = tuple(traces)
        self.policy = policy

    @classmethod
    def from_tasks(
        cls,
        tasks: Sequence[NarrowTask],
        *,
        policy: str = "size_topology",
    ) -> "TraceDataset":
        """Build the dataset from narrow tasks using the deterministic reference solver."""
        return cls([build_trace(task) for task in tasks], policy=policy)

    def __len__(self) -> int:
        return len(self.traces)

    def by_task_type(self) -> dict[str, list[AlgorithmTrace]]:
        grouped: dict[str, list[AlgorithmTrace]] = {}
        for trace in self.traces:
            grouped.setdefault(trace.task.task_type, []).append(trace)
        return grouped

    def summary(self) -> dict[str, object]:
        """Return dataset-level manifest metadata."""
        grouped = self.by_task_type()
        return {
            "n_traces": len(self.traces),
            "task_types": sorted(grouped),
            "split_policy": self.policy,
            "implementation_versions": sorted({
                t.implementation_version for t in self.traces
            }),
            "seeds": sorted({t.seed for t in self.traces}),
            "reference_version": NAR_REFERENCE_VERSION,
        }


__all__ = [
    "EDGE_FEATURE_DIM", "NODE_FEATURE_DIM", "AlgorithmTrace", "TraceDataset", "build_trace",
]
