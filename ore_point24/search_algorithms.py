"""Phase V, Step 24 - integrate search, planning and predictive world models.

Select the *smallest* suitable search or planning method: Dijkstra for
non-negative edge costs, D* Lite for incremental replanning, RRT for suitable
continuous geometric spaces, minimax with alpha-beta for adversarial search,
and dynamic programming where subproblem structure permits reuse.  A plan is
returned with preconditions, expected outcomes, costs and failure
alternatives; execution must re-check safety constraints against the actual
state before each consequential action.
"""
from __future__ import annotations

import heapq
import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Hashable, Iterable, Mapping, Sequence

from .core import new_id, sha256_digest, utc_now_iso

SEARCH_SCHEMA = "ore.point24.search.v1"
INFINITY = float("inf")
Node = Hashable


@dataclass(frozen=True, slots=True)
class PlanStep:
    """One action in a candidate plan."""

    action: str
    state: Node
    cost: float
    preconditions: tuple[str, ...] = ()
    expected_outcome: str = ""
    failure_alternative: str | None = None


@dataclass(frozen=True, slots=True)
class CandidatePlan:
    """A candidate plan with total cost, assumptions and safety re-checks."""

    objective: str
    steps: tuple[PlanStep, ...]
    total_cost: float
    method: str
    assumptions: tuple[str, ...] = ()
    model_uncertainty: str | None = None
    horizon: int | None = None
    plan_id: str = field(default_factory=lambda: new_id("plan"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SEARCH_SCHEMA,
            "plan_id": self.plan_id,
            "objective": self.objective,
            "method": self.method,
            "total_cost": self.total_cost,
            "assumptions": list(self.assumptions),
            "model_uncertainty": self.model_uncertainty,
            "horizon": self.horizon,
            "steps": [
                {
                    "action": step.action, "state": str(step.state), "cost": step.cost,
                    "preconditions": list(step.preconditions),
                    "expected_outcome": step.expected_outcome,
                    "failure_alternative": step.failure_alternative,
                }
                for step in self.steps
            ],
        }


def dijkstra(
    graph: Mapping[Node, Mapping[Node, float]],
    source: Node,
    target: Node,
) -> tuple[list[Node], float]:
    """Shortest path with non-negative edge costs (assumption is enforced)."""
    for node, edges in graph.items():
        for neighbour, cost in edges.items():
            if cost < 0:
                raise ValueError(
                    f"Dijkstra requires non-negative edge costs; edge "
                    f"{node!r}->{neighbour!r} has {cost}"
                )
    distances: dict[Node, float] = {source: 0.0}
    previous: dict[Node, Node] = {}
    visited: set[Node] = set()
    queue: list[tuple[float, int, Node]] = [(0.0, 0, source)]
    counter = 0
    while queue:
        distance, _, node = heapq.heappop(queue)
        if node in visited:
            continue
        visited.add(node)
        if node == target:
            break
        for neighbour, cost in graph.get(node, {}).items():
            if cost < 0:
                raise ValueError("Dijkstra requires non-negative edge costs")
            candidate = distance + cost
            if candidate < distances.get(neighbour, INFINITY):
                distances[neighbour] = candidate
                previous[neighbour] = node
                counter += 1
                heapq.heappush(queue, (candidate, counter, neighbour))
    if target not in distances:
        return [], INFINITY
    path = [target]
    while path[-1] != source:
        path.append(previous[path[-1]])
    path.reverse()
    return path, distances[target]


class DStarLite:
    """Incremental replanning with D* Lite (simplified, grid-world first version).

    When edges change after the initial search, only the affected region is
    reprocessed instead of replanning from scratch.  The implementation keeps
    the standard key ordering ``k(s) = [min(g(s), rhs(s)) + h(s_start, s),
    min(g(s), rhs(s))]``.
    """

    def __init__(self, neighbours: Callable[[Node], Iterable[tuple[Node, float]]], start: Node, goal: Node) -> None:
        self._neighbours = neighbours
        self.start = start
        self.goal = goal
        self.g: dict[Node, float] = {}
        self.rhs: dict[Node, float] = {goal: 0.0}
        self.queue: list[tuple[tuple[float, float], int, Node]] = []
        self._counter = 0
        self.km = 0.0
        self._insert(goal, self._key(goal))

    def _h(self, node: Node) -> float:
        return 0.0  # admissible default; callers may subclass for grids

    def _key(self, node: Node) -> tuple[float, float]:
        g = self.g.get(node, INFINITY)
        rhs = self.rhs.get(node, INFINITY)
        min_gr = min(g, rhs)
        return (min_gr + self._h(node) + self.km, min_gr)

    def _insert(self, node: Node, key: tuple[float, float]) -> None:
        self._counter += 1
        heapq.heappush(self.queue, (key, self._counter, node))

    def _update_vertex(self, node: Node) -> None:
        if node != self.goal:
            costs = [
                cost + self.g.get(neighbour, INFINITY)
                for neighbour, cost in self._neighbours(node)
            ]
            self.rhs[node] = min(costs) if costs else INFINITY
        g = self.g.get(node, INFINITY)
        rhs = self.rhs.get(node, INFINITY)
        in_queue = any(item[2] == node for item in self.queue)
        if g != rhs and not in_queue:
            self._insert(node, self._key(node))

    def _top_key(self) -> tuple[float, float] | None:
        return self.queue[0][0] if self.queue else None

    def _compute_shortest_path(self, max_iterations: int = 10_000) -> None:
        iterations = 0
        while self.queue and iterations < max_iterations:
            top = self._top_key()
            start_key = self._key(self.start)
            start_g = self.g.get(self.start, INFINITY)
            start_rhs = self.rhs.get(self.start, INFINITY)
            if top is None or (top >= start_key and start_g == start_rhs):
                break
            _, _, node = heapq.heappop(self.queue)
            node_key = self._key(node)
            if top < node_key:
                self._insert(node, node_key)
                continue
            g = self.g.get(node, INFINITY)
            rhs = self.rhs.get(node, INFINITY)
            if g > rhs:
                self.g[node] = rhs
            else:
                self.g[node] = INFINITY
                self._update_vertex(node)
            for neighbour, _ in self._neighbours(node):
                self._update_vertex(neighbour)
            iterations += 1

    def plan(self) -> tuple[list[Node], float]:
        self._compute_shortest_path()
        cost = self.g.get(self.start, INFINITY)
        if not math.isfinite(cost):
            return [], INFINITY
        path = [self.start]
        current = self.start
        guard = 0
        while current != self.goal and guard < 10_000:
            guard += 1
            options = [
                (cost_edge + self.g.get(neighbour, INFINITY), neighbour)
                for neighbour, cost_edge in self._neighbours(current)
            ]
            if not options:
                return [], INFINITY
            _, current = min(options)
            path.append(current)
        return path, self.g.get(self.start, INFINITY)

    def edge_changed(self, node: Node) -> None:
        """Notify an edge-cost change; replans incrementally."""
        self.km += self._h(self.start)
        self._update_vertex(node)
        for neighbour, _ in self._neighbours(node):
            self._update_vertex(neighbour)
        self._compute_shortest_path()


def rrt(
    *,
    sample_space: tuple[tuple[float, float], tuple[float, float]],
    start: tuple[float, float],
    goal: tuple[float, float],
    is_collision_free: Callable[[tuple[float, float], tuple[float, float]], bool],
    step_size: float = 0.25,
    goal_tolerance: float = 0.3,
    max_iterations: int = 2_000,
    seed: int = 7,
) -> tuple[list[tuple[float, float]], bool]:
    """Rapidly-exploring random tree over a 2-D continuous space.

    Returns the path and whether the goal was reached within tolerance and
    the iteration budget.  Suitable for continuous geometric spaces where
    edge costs are locally metric; collision checking is delegated to the
    caller-supplied predicate.
    """
    rng = random.Random(seed)
    (x_min, x_max), (y_min, y_max) = sample_space
    nodes: list[tuple[float, float]] = [tuple(start)]
    parents: dict[int, int] = {0: 0}
    reached = False
    for _ in range(max_iterations):
        if rng.random() < 0.1:
            sample = tuple(goal)
        else:
            sample = (rng.uniform(x_min, x_max), rng.uniform(y_min, y_max))
        nearest_index = min(
            range(len(nodes)),
            key=lambda index: (nodes[index][0] - sample[0]) ** 2 + (nodes[index][1] - sample[1]) ** 2,
        )
        nearest = nodes[nearest_index]
        dx = sample[0] - nearest[0]
        dy = sample[1] - nearest[1]
        norm = math.hypot(dx, dy)
        if norm == 0:
            continue
        new = (nearest[0] + step_size * dx / norm, nearest[1] + step_size * dy / norm)
        if not (x_min <= new[0] <= x_max and y_min <= new[1] <= y_max):
            continue
        if not is_collision_free(nearest, new):
            continue
        nodes.append(new)
        parents[len(nodes) - 1] = nearest_index
        if math.hypot(new[0] - goal[0], new[1] - goal[1]) <= goal_tolerance:
            reached = True
            break
    if not reached:
        return [tuple(start)], False
    path_index = len(nodes) - 1
    path = [nodes[path_index]]
    while path_index != 0:
        path_index = parents[path_index]
        path.append(nodes[path_index])
    path.reverse()
    return path, True


@dataclass(frozen=True, slots=True)
class AdversarialResult:
    """The minimax value and best move for the maximizing player."""

    value: float
    best_action: Any | None
    nodes_evaluated: int


def minimax(
    *,
    actions: Callable[[tuple[int, list[Any]],], Sequence[Any]],
    transition: Callable[[tuple[int, list[Any]], Any], tuple[int, list[Any]]],
    utility: Callable[[tuple[int, list[Any]]], float],
    terminal: Callable[[tuple[int, list[Any]]], bool],
    depth: int,
    state: tuple[int, list[Any]],
    use_alpha_beta: bool = True,
) -> AdversarialResult:
    """Adversarial search (minimax with optional alpha-beta pruning).

    ``state`` is ``(player, payload)`` where player 0 maximises and player 1
    minimises.  Alpha-beta returns the same root value as plain minimax while
    pruning provably irrelevant branches.
    """
    counter = {"nodes": 0}

    def recurse(current: tuple[int, list[Any]], remaining: int,
                alpha: float, beta: float) -> tuple[float, Any]:
        counter["nodes"] += 1
        if terminal(current) or remaining == 0:
            return utility(current), None
        maximizing = current[0] == 0
        best_value = -INFINITY if maximizing else INFINITY
        best_action = None
        for action in actions(current):
            value, _ = recurse(transition(current, action), remaining - 1, alpha, beta)
            if maximizing:
                if value > best_value:
                    best_value, best_action = value, action
                alpha = max(alpha, best_value)
            else:
                if value < best_value:
                    best_value, best_action = value, action
                beta = min(beta, best_value)
            if use_alpha_beta and beta <= alpha:
                break
        return best_value, best_action

    value, action = recurse(state, depth, -INFINITY, INFINITY)
    return AdversarialResult(value=value, best_action=action, nodes_evaluated=counter["nodes"])


def dynamic_programming(
    *,
    stages: int,
    state_list: Sequence[Any],
    value_terminal: Callable[[Any], float],
    transition_cost: Callable[[int, Any, Any], float],
) -> tuple[float, list[Any]]:
    """Backward dynamic programming over discrete stages.

    ``transition_cost(stage, from_state, to_state)`` must satisfy the
    principle of optimality; subproblems are reused rather than recomputed.
    """
    if stages < 1:
        raise ValueError("stages must be at least one")
    best_value: dict[Any, float] = {}
    best_next: dict[int, dict[Any, Any]] = {}
    for state in state_list:
        best_value[state] = value_terminal(state)
    for stage in range(stages - 1, -1, -1):
        stage_next: dict[Any, Any] = {}
        new_value: dict[Any, float] = {}
        for state in state_list:
            options = [
                (transition_cost(stage, state, nxt) + best_value[nxt], nxt)
                for nxt in state_list
            ]
            cost, nxt = min(options, key=lambda item: (item[0], str(item[1])))
            new_value[state] = cost
            stage_next[state] = nxt
        best_value = new_value
        best_next[stage] = stage_next
    initial = state_list[0]
    total = best_value[initial]
    trajectory = [initial]
    current = initial
    for stage in range(stages):
        current = best_next[stage][current]
        trajectory.append(current)
    return total, trajectory


@dataclass(frozen=True, slots=True)
class HTNTask:
    """One hierarchical task-network element (compound or primitive)."""

    name: str
    kind: str  # primitive | compound
    subtasks: tuple[str, ...] = ()
    action: str | None = None

    def __post_init__(self) -> None:
        if self.kind == "compound" and not self.subtasks:
            raise ValueError("compound tasks require subtasks")
        if self.kind == "primitive" and not self.action:
            raise ValueError("primitive tasks require an action")


def htn_decompose(
    tasks: Mapping[str, HTNTask],
    root: str,
    *,
    max_depth: int = 16,
) -> list[str]:
    """Decompose an HTN root task into a sequence of primitive actions.

    Uses declared task structure only; the decomposition fails loudly on
    unknown task names or cyclic decompositions rather than inventing steps.
    """
    result: list[str] = []
    visited: set[str] = set()

    def expand(name: str, depth: int) -> None:
        if depth > max_depth:
            raise ValueError(f"HTN decomposition exceeded depth {max_depth}")
        if name in visited and tasks[name].kind == "compound":
            raise ValueError(f"cyclic HTN decomposition at task {name!r}")
        task = tasks.get(name)
        if task is None:
            raise KeyError(f"unknown HTN task: {name!r}")
        if task.kind == "primitive":
            result.append(task.action or name)
            return
        visited.add(name)
        for subtask in task.subtasks:
            expand(subtask, depth + 1)
        visited.discard(name)

    expand(root, 0)
    return result


def mpc_rollout(
    *,
    initial_state: Sequence[float],
    dynamics: Callable[[Sequence[float], int], Sequence[float]],
    cost: Callable[[Sequence[float], int], float],
    action_set: Sequence[int],
    horizon: int,
) -> tuple[list[int], list[float], str]:
    """Model-predictive control with an explicit model-uncertainty caveat.

    Returns (actions, predicted_costs, model_uncertainty).  MPC is applicable
    only when a sufficiently validated predictive model exists; the caller
    must supply the model, and the returned uncertainty note records that the
    horizon and model error bound the guarantee.
    """
    state = tuple(initial_state)
    actions: list[int] = []
    costs: list[float] = []
    for _ in range(horizon):
        best_action = None
        best_cost = INFINITY
        for action in action_set:
            next_state = dynamics(state, action)
            step_cost = cost(next_state, action)
            if step_cost < best_cost:
                best_cost, best_action = step_cost, action
        if best_action is None:
            break
        state = dynamics(state, best_action)
        actions.append(best_action)
        costs.append(best_cost)
    return actions, costs, (
        "greedy-horizon MPC; guarantee bounded by predictive-model validity "
        f"and horizon={horizon}"
    )


__all__ = [
    "SEARCH_SCHEMA",
    "AdversarialResult",
    "CandidatePlan",
    "DStarLite",
    "HTNTask",
    "PlanStep",
    "dijkstra",
    "dynamic_programming",
    "htn_decompose",
    "minimax",
    "mpc_rollout",
    "rrt",
]
