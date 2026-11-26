"""Deterministic narrow-task generators and trusted reference algorithms (ORE Point 8 steps 41-42).

Neural algorithmic reasoning is restricted to *narrow* tasks: graph traversal (BFS),
shortest paths (Bellman-Ford relaxation), connectivity (label propagation), minimum
spanning tree (Kruskal) and sorting. Unrestricted algorithm learning is explicitly out
of scope. Every generated task carries its seed, parameters and implementation version
so that traces are reproducible, and reference outputs are produced only by these
deterministic algorithms - never by a learned model.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

import numpy as np

#: Narrow tasks permitted for the NAR research track (step 41).
NAR_TASK_TYPES = (
    "graph_traversal_bfs",
    "shortest_path",
    "connectivity",
    "minimum_spanning_tree",
    "sorting",
)

#: Graph topologies a generator can emit; "grid" is reserved for OOD evaluation.
TOPOLOGIES = ("chain", "ring", "star", "binary_tree", "grid", "random")

#: Training topologies (grid stays unseen for out-of-distribution tests, step 45).
TRAINING_TOPOLOGIES = tuple(t for t in TOPOLOGIES if t != "grid")

#: Deterministic implementation version pinned for trace reproducibility (step 42).
NAR_REFERENCE_VERSION = "nar-deterministic-reference-1.0.0"

_WEIGHT_MAX = 9.0


def _edge_key(u: int, v: int) -> tuple[int, int]:
    return (u, v) if u <= v else (v, u)


@dataclass(frozen=True, slots=True)
class NarrowTask:
    """One reproducible algorithmic task: input structure, parameters, seed and version."""

    task_type: str
    seed: int
    n_nodes: int
    topology: str
    edges: tuple[tuple[int, int, float], ...] = ()
    sequence: tuple[float, ...] = ()
    source_node: int = 0
    implementation_version: str = NAR_REFERENCE_VERSION
    task_id: str = ""

    def __post_init__(self) -> None:
        if self.task_type not in NAR_TASK_TYPES:
            raise ValueError(f"task_type must be one of {list(NAR_TASK_TYPES)}")
        if self.task_type == "sorting":
            if not self.sequence:
                raise ValueError("sorting tasks require a non-empty sequence")
        else:
            if not self.edges:
                raise ValueError(f"{self.task_type} tasks require edges")
        if self.n_nodes < 2:
            raise ValueError("n_nodes must be at least 2")
        if self.topology not in TOPOLOGIES:
            raise ValueError(f"topology must be one of {list(TOPOLOGIES)}")
        for u, v, _w in self.edges:
            if not 0 <= u < self.n_nodes or not 0 <= v < self.n_nodes:
                raise ValueError("edge endpoints must reference existing nodes")
        if not self.task_id:
            digest = sha256(
                "|".join((
                    self.task_type, str(self.seed), str(self.n_nodes), self.topology,
                    ",".join(f"{u}-{v}-{w:.4f}" for u, v, w in self.edges),
                    ",".join(f"{v:.4f}" for v in self.sequence),
                )).encode("utf-8"),
            ).hexdigest()
            object.__setattr__(self, "task_id", f"nar-task:{digest[:24]}")

    def adjacency_undirected(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (sources, targets, weights) with both directions of every edge."""
        pairs = []
        for u, v, w in self.edges:
            pairs.append((u, v, w))
            if u != v:
                pairs.append((v, u, w))
        sources = np.array([p[0] for p in pairs], dtype=np.int64)
        targets = np.array([p[1] for p in pairs], dtype=np.int64)
        weights = np.array([p[2] for p in pairs], dtype=np.float64)
        return sources, targets, weights


class TaskGenerator:
    """Generate narrow tasks deterministically from a seed (step 41-42).

    ``generate`` never consults a learned model: task inputs, parameters and the seed
    fully determine the task identity, and reference algorithms run on the same inputs.
    """

    def __init__(self, *, implementation_version: str = NAR_REFERENCE_VERSION) -> None:
        self.implementation_version = implementation_version

    def generate(
        self,
        task_type: str,
        *,
        n_nodes: int = 12,
        seed: int = 0,
        topology: str = "random",
        edge_probability: float = 0.25,
    ) -> NarrowTask:
        """Build one task of the requested type, size, topology and seed."""
        if task_type not in NAR_TASK_TYPES:
            raise ValueError(f"task_type must be one of {list(NAR_TASK_TYPES)}")
        if topology not in TOPOLOGIES:
            raise ValueError(f"topology must be one of {list(TOPOLOGIES)}")
        if not 0.0 < edge_probability <= 1.0:
            raise ValueError("edge_probability must be in (0, 1]")
        rng = np.random.default_rng(seed)
        if task_type == "sorting":
            values = rng.uniform(0.0, 1.0, max(2, n_nodes))
            return NarrowTask(
                task_type=task_type,
                seed=int(seed),
                n_nodes=int(values.size),
                topology=topology,
                sequence=tuple(round(float(v), 6) for v in values),
                implementation_version=self.implementation_version,
            )
        edges = self._build_edges(rng, topology, n_nodes, edge_probability)
        weights = rng.integers(1, int(_WEIGHT_MAX) + 1, size=len(edges)).astype(np.float64)
        weighted = tuple(
            (int(u), int(v), float(w)) for (u, v), w in zip(edges, weights, strict=True)
        )
        return NarrowTask(
            task_type=task_type,
            seed=int(seed),
            n_nodes=int(n_nodes),
            topology=topology,
            edges=weighted,
            implementation_version=self.implementation_version,
        )

    # ------------------------------------------------------------------ topologies

    def _build_edges(
        self,
        rng: np.random.Generator,
        topology: str,
        n_nodes: int,
        edge_probability: float,
    ) -> list[tuple[int, int]]:
        if topology == "chain":
            return [(i, i + 1) for i in range(n_nodes - 1)]
        if topology == "ring":
            return [(i, (i + 1) % n_nodes) for i in range(n_nodes)]
        if topology == "star":
            return [(0, i) for i in range(1, n_nodes)]
        if topology == "binary_tree":
            return [((i - 1) // 2, i) for i in range(1, n_nodes)]
        if topology == "grid":
            side = max(2, int(np.sqrt(n_nodes)))
            nodes = side * side
            if nodes != n_nodes:
                raise ValueError(f"grid topology requires a square node count (got {n_nodes})")
            pairs: list[tuple[int, int]] = []
            for r in range(side):
                for c in range(side):
                    node = r * side + c
                    if c + 1 < side:
                        pairs.append((node, node + 1))
                    if r + 1 < side:
                        pairs.append((node, node + side))
            return pairs
        if topology == "random":
            pairs = []
            for u in range(n_nodes):
                for v in range(u + 1, n_nodes):
                    if bool(rng.random() < edge_probability):
                        pairs.append((u, v))
            return pairs if pairs else [(0, 1)]
        raise ValueError(f"unknown topology: {topology}")  # pragma: no cover - guarded above


# ------------------------------------------------------------------ reference algorithms


@dataclass(frozen=True, slots=True)
class ReferenceOutput:
    """Deterministic reference solution for one narrow task (step 42).

    ``step_states`` are the intermediate algorithm states (one row per step) that a
    trained model must imitate; ``final`` is the finished output. All values come from
    the trusted reference implementation, never from a learned model.
    """

    task_id: str
    step_states: tuple[np.ndarray, ...]
    final: np.ndarray
    node_output: np.ndarray
    edge_output: np.ndarray
    termination_step: int
    implementation_version: str


def reference_solve(task: NarrowTask) -> ReferenceOutput:
    """Run the trusted deterministic algorithm for the task type (step 42)."""
    if task.task_type == "graph_traversal_bfs":
        return _reference_bfs(task)
    if task.task_type == "shortest_path":
        return _reference_shortest_path(task)
    if task.task_type == "connectivity":
        return _reference_connectivity(task)
    if task.task_type == "minimum_spanning_tree":
        return _reference_mst(task)
    if task.task_type == "sorting":
        return _reference_sorting(task)
    raise ValueError(f"unsupported task type: {task.task_type}")  # pragma: no cover


def _reference_bfs(task: NarrowTask) -> ReferenceOutput:
    visited = np.zeros(task.n_nodes, dtype=np.float64)
    visited[task.source_node] = 1.0
    states = [visited.copy()]
    adjacency: dict[int, list[int]] = {i: [] for i in range(task.n_nodes)}
    for u, v, _w in task.edges:
        adjacency[u].append(v)
        adjacency[v].append(u)
    frontier = [task.source_node]
    for _step in range(task.n_nodes):
        if not frontier:
            break
        next_frontier: list[int] = []
        for node in frontier:
            for neighbor in adjacency[node]:
                if visited[neighbor] == 0.0:
                    visited[neighbor] = 1.0
                    next_frontier.append(neighbor)
        frontier = next_frontier
        states.append(visited.copy())
    final = states[-1]
    return ReferenceOutput(
        task_id=task.task_id,
        step_states=tuple(states),
        final=final,
        node_output=final.copy(),
        edge_output=np.zeros(len(task.edges), dtype=np.float64),
        termination_step=len(states) - 1,
        implementation_version=task.implementation_version,
    )


def _reference_shortest_path(task: NarrowTask) -> ReferenceOutput:
    cap = float(task.n_nodes * (_WEIGHT_MAX + 1.0))
    distances = np.full(task.n_nodes, cap, dtype=np.float64)
    distances[task.source_node] = 0.0
    states = [distances.copy()]
    adjacency: dict[int, list[tuple[int, float]]] = {i: [] for i in range(task.n_nodes)}
    for u, v, w in task.edges:
        adjacency[u].append((v, w))
        adjacency[v].append((u, w))
    for _step in range(task.n_nodes):
        changed = False
        new = distances.copy()
        for node, dist in enumerate(distances):
            if dist >= cap:
                continue
            for neighbor, weight in adjacency[node]:
                candidate = dist + weight
                if candidate < new[neighbor]:
                    new[neighbor] = candidate
                    changed = True
        distances = new
        states.append(distances.copy())
        if not changed:
            break
    return ReferenceOutput(
        task_id=task.task_id,
        step_states=tuple(states),
        final=distances,
        node_output=distances.copy(),
        edge_output=np.zeros(len(task.edges), dtype=np.float64),
        termination_step=len(states) - 1,
        implementation_version=task.implementation_version,
    )


def _reference_connectivity(task: NarrowTask) -> ReferenceOutput:
    labels = np.arange(task.n_nodes, dtype=np.float64)
    states = [labels.copy()]
    adjacency: dict[int, list[int]] = {i: [] for i in range(task.n_nodes)}
    for u, v, _w in task.edges:
        adjacency[u].append(v)
        adjacency[v].append(u)
    for _step in range(task.n_nodes):
        changed = False
        new = labels.copy()
        for node in range(task.n_nodes):
            for neighbor in adjacency[node]:
                if new[node] > labels[neighbor]:
                    new[node] = labels[neighbor]
                    changed = True
        labels = new
        states.append(labels.copy())
        if not changed:
            break
    return ReferenceOutput(
        task_id=task.task_id,
        step_states=tuple(states),
        final=labels,
        node_output=labels.copy(),
        edge_output=np.zeros(len(task.edges), dtype=np.float64),
        termination_step=len(states) - 1,
        implementation_version=task.implementation_version,
    )


def _reference_mst(task: NarrowTask) -> ReferenceOutput:
    order = sorted(range(len(task.edges)), key=lambda i: (task.edges[i][2], i))
    parent = list(range(task.n_nodes))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    in_tree = np.zeros(len(task.edges), dtype=np.float64)
    for index in order:
        u, v, _w = task.edges[index]
        root_u, root_v = find(u), find(v)
        if root_u != root_v:
            parent[root_u] = root_v
            in_tree[index] = 1.0
    return ReferenceOutput(
        task_id=task.task_id,
        step_states=(in_tree.copy(),),
        final=in_tree,
        node_output=np.zeros(task.n_nodes, dtype=np.float64),
        edge_output=in_tree.copy(),
        termination_step=0,
        implementation_version=task.implementation_version,
    )


def _reference_sorting(task: NarrowTask) -> ReferenceOutput:
    values = np.array(task.sequence, dtype=np.float64)
    order = np.argsort(values, kind="stable")
    ranks = np.zeros(values.size, dtype=np.float64)
    ranks[order] = np.arange(values.size, dtype=np.float64)
    final = values[order]
    return ReferenceOutput(
        task_id=task.task_id,
        step_states=(ranks.copy(),),
        final=final,
        node_output=ranks.copy(),
        edge_output=np.zeros(0, dtype=np.float64),
        termination_step=0,
        implementation_version=task.implementation_version,
    )


__all__ = [
    "NAR_TASK_TYPES", "NAR_REFERENCE_VERSION", "TOPOLOGIES", "TRAINING_TOPOLOGIES",
    "NarrowTask", "ReferenceOutput", "TaskGenerator", "reference_solve",
]
