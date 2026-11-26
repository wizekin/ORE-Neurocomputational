"""Cost-aware search over legal plans: A*, weighted A*, and bounded beam search.

Successor generation respects the partial-order constraint graph: only nodes whose
predecessors have executed can expand, and each expansion selects one eligible
provider operator for the node's capability whose preconditions hold. Duplicate
states are detected by deterministic state hashing, dominance prunes worse paths
to the same state, and node/time/cost budgets are enforced. An optimality claim
is emitted only when an admissible heuristic drove a complete, untruncated A*
search. Beam search is explicitly incomplete: it records every pruned branch.
"""
from __future__ import annotations

import heapq
import math
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .contracts import (
    GoalSpec,
    OperatorSpec,
    PlannerBudget,
    PlanState,
    SearchRecord,
)
from .cost import CostModel, RemainingCostHeuristic, UnmetConditionHeuristic
from .partial_order import PartialOrderPlan


@dataclass(frozen=True, slots=True)
class SearchStats:
    """Report search effort and pruning decisions for auditability."""

    expanded: int
    generated: int
    duplicates_pruned: int
    dominance_pruned: int
    beam_pruned: int
    budget_exhausted: bool

    def to_dict(self) -> dict:
        """Return the search statistics."""
        return {
            "expanded": self.expanded,
            "generated": self.generated,
            "duplicates_pruned": self.duplicates_pruned,
            "dominance_pruned": self.dominance_pruned,
            "beam_pruned": self.beam_pruned,
            "budget_exhausted": self.budget_exhausted,
        }


@dataclass(frozen=True, slots=True)
class SearchOutcome:
    """Hold the best discovered plan and the audit trail that produced it."""

    operator_sequence: tuple[tuple[str, str], ...]
    total_cost: float
    goal_achieved: bool
    complete: bool
    stop_reason: str
    optimality_claim: bool
    records: tuple[SearchRecord, ...]
    stats: SearchStats

    def __post_init__(self) -> None:
        """Normalize the operator sequence."""
        object.__setattr__(
            self, "operator_sequence",
            tuple((str(node), str(operator)) for node, operator in self.operator_sequence),
        )

    def to_dict(self) -> dict:
        """Return a detached, JSON-compatible outcome record."""
        return {
            "operator_sequence": [
                {"node_id": node, "operator_id": operator}
                for node, operator in self.operator_sequence
            ],
            "total_cost": self.total_cost,
            "goal_achieved": self.goal_achieved,
            "complete": self.complete,
            "stop_reason": self.stop_reason,
            "optimality_claim": self.optimality_claim,
            "records": [item.to_dict() for item in self.records],
            "stats": self.stats.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class _SearchNode:
    """Hold one search frontier entry without heap-comparable payload fields."""

    g_cost: float
    facts: frozenset[str]
    executed: frozenset[str]
    sequence: tuple[tuple[str, str], ...]
    state_hash: str
    parent_hash: str | None
    h_cost: float = field(compare=False, default=0.0)


class SearchEngine:
    """Search legal plans with A*, weighted A*, or bounded beam search."""

    def __init__(
        self,
        cost_model: CostModel,
        heuristic: RemainingCostHeuristic | None = None,
        *,
        budget: PlannerBudget | None = None,
        mode: str = "astar",
        weight: float = 1.0,
    ) -> None:
        """Configure the search mode, heuristic, and enforced budgets."""
        if mode not in {"astar", "weighted_astar", "beam"}:
            raise ValueError("mode must be one of astar, weighted_astar, beam")
        if not math.isfinite(weight) or weight < 1.0:
            raise ValueError("weight must be >= 1.0 for weighted A*")
        self.cost_model = cost_model
        self.heuristic = heuristic or UnmetConditionHeuristic()
        self.budget = budget or PlannerBudget()
        self.mode = mode
        self.weight = weight

    @property
    def heuristic_version(self) -> str:
        """Return the documented heuristic version for decision records."""
        return self.heuristic.version

    def _ready_nodes(
        self, plan: PartialOrderPlan, executed: frozenset[str]
    ) -> tuple[str, ...]:
        """Return capability nodes whose predecessors have all executed."""
        ready = []
        for node in plan.nodes:
            if node.node_id in executed:
                continue
            predecessors = {
                edge.predecessor for edge in plan.edges
                if edge.successor == node.node_id
            }
            if predecessors <= executed:
                ready.append(node.node_id)
        return tuple(sorted(ready))

    def _applicable_operators(
        self,
        node_id: str,
        plan: PartialOrderPlan,
        facts: frozenset[str],
        operators: Mapping[str, OperatorSpec],
    ) -> tuple[OperatorSpec, ...]:
        """Return eligible provider operators whose preconditions hold for one node."""
        node = plan.node(node_id)
        return tuple(
            operator for operator in sorted(
                (item for item in operators.values()
                 if item.module_capability == node.task.task_type),
                key=lambda item: item.operator_id,
            )
            if all(precondition.satisfied_by(facts) for precondition in operator.preconditions)
        )

    def _state_for(
        self, initial: PlanState, facts: frozenset[str], executed: frozenset[str]
    ) -> PlanState:
        """Derive one search state from the initial binding, facts, and executed nodes."""
        return PlanState(
            workspace_revision=initial.workspace_revision,
            achieved_facts=frozenset(facts),
            pending_tasks=tuple(sorted(executed)),
            resources=dict(initial.resources),
            hypotheses=dict(initial.hypotheses),
            plan_version=initial.plan_version,
        )

    def _successors(
        self,
        entry: _SearchNode,
        plan: PartialOrderPlan,
        goal: GoalSpec,
        operators: Mapping[str, OperatorSpec],
        initial_state: PlanState,
        seen: dict[str, float],
        counters: dict[str, int],
    ) -> tuple[list[_SearchNode], int, int]:
        """Generate applicable successor nodes with duplicate and dominance pruning."""
        successors: list[_SearchNode] = []
        duplicates = 0
        dominance = 0
        for node_id in self._ready_nodes(plan, entry.executed):
            for operator in self._applicable_operators(
                node_id, plan, entry.facts, operators,
            ):
                new_facts = _apply_effects(entry.facts, operator)
                new_executed = entry.executed | {node_id}
                new_sequence = entry.sequence + ((node_id, operator.operator_id),)
                new_state = self._state_for(initial_state, new_facts, new_executed)
                new_g = entry.g_cost + self.cost_model.operator_cost(operator)
                state_hash = new_state.state_key()
                counters["generated"] += 1
                if state_hash in seen and seen[state_hash] <= new_g:
                    if seen[state_hash] == new_g:
                        duplicates += 1
                    else:
                        dominance += 1
                    continue
                seen[state_hash] = new_g
                successors.append(_SearchNode(
                    g_cost=new_g,
                    facts=new_facts,
                    executed=new_executed,
                    sequence=new_sequence,
                    state_hash=state_hash,
                    parent_hash=entry.state_hash,
                    h_cost=self.heuristic.estimate(
                        new_state, goal, operators, self.cost_model,
                    ),
                ))
        return successors, duplicates, dominance

    def _goal_reached(self, goal: GoalSpec, facts: frozenset[str], threshold: float) -> bool:
        """Apply the goal test: critical hard gates plus the quality threshold."""
        progress = self.cost_model.goal_progress(goal, facts)
        return progress.critical_satisfied and progress.score >= threshold

    def search(
        self,
        plan: PartialOrderPlan,
        goal: GoalSpec,
        operators: Mapping[str, OperatorSpec],
        initial_state: PlanState,
        *,
        quality_threshold: float = 1.0,
    ) -> SearchOutcome:
        """Run the configured search and return the best outcome with its audit trail."""
        if self.mode == "beam":
            return self._beam_search(
                plan, goal, operators, initial_state, quality_threshold,
            )
        started = time.monotonic()
        records: list[SearchRecord] = []
        counters = {"generated": 0}
        seen: dict[str, float] = {}
        expanded = 0
        duplicates_total = 0
        dominance_total = 0
        budget_exhausted = False
        best: _SearchNode | None = None

        initial_hash = initial_state.state_key()
        seen[initial_hash] = 0.0
        initial_node = _SearchNode(
            g_cost=0.0,
            facts=frozenset(initial_state.achieved_facts),
            executed=frozenset(),
            sequence=(),
            state_hash=initial_hash,
            parent_hash=None,
            h_cost=self.heuristic.estimate(
                initial_state, goal, operators, self.cost_model,
            ),
        )
        counter = 0
        heap: list[tuple[float, int, _SearchNode]] = [
            (self._priority(initial_node), counter, initial_node),
        ]

        while heap:
            if expanded >= self.budget.max_search_nodes:
                budget_exhausted = True
                break
            if time.monotonic() - started > self.budget.max_time_seconds:
                budget_exhausted = True
                break
            _, _, entry = heapq.heappop(heap)
            if entry.g_cost > seen.get(entry.state_hash, math.inf):
                continue
            if self._goal_reached(goal, entry.facts, quality_threshold):
                if best is None or entry.g_cost < best.g_cost:
                    best = entry
                if self.heuristic.admissible and self.heuristic.consistent \
                        and self.weight == 1.0:
                    break  # first goal pop is optimal with a consistent heuristic
                continue  # otherwise exhaust the frontier; the claim stays sound
            if entry.g_cost > self.budget.max_plan_cost:
                budget_exhausted = True
                continue
            if len(entry.sequence) >= self.budget.max_depth:
                continue
            expanded += 1
            records.append(SearchRecord(
                state_hash=entry.state_hash,
                g_cost=entry.g_cost,
                h_cost=entry.h_cost,
                parent=entry.parent_hash,
                chosen_operator=entry.sequence[-1][1] if entry.sequence else None,
                depth=len(entry.sequence),
            ))
            successors, duplicates, dominance = self._successors(
                entry, plan, goal, operators, initial_state, seen, counters,
            )
            duplicates_total += duplicates
            dominance_total += dominance
            for successor in successors:
                counter += 1
                heapq.heappush(heap, (self._priority(successor), counter, successor))

        stats = SearchStats(
            expanded=expanded,
            generated=counters["generated"],
            duplicates_pruned=duplicates_total,
            dominance_pruned=dominance_total,
            beam_pruned=0,
            budget_exhausted=budget_exhausted,
        )
        if best is not None:
            optimality = bool(
                self.mode == "astar" and self.heuristic.admissible
                and self.weight == 1.0 and not budget_exhausted
            )
            return SearchOutcome(
                operator_sequence=best.sequence,
                total_cost=best.g_cost,
                goal_achieved=True,
                complete=True,
                stop_reason="verified_success_conditions",
                optimality_claim=optimality,
                records=tuple(records),
                stats=stats,
            )
        fallback = self._best_partial(heap, initial_state)
        return SearchOutcome(
            operator_sequence=fallback[0],
            total_cost=fallback[1],
            goal_achieved=False,
            complete=False,
            stop_reason=(
                "search_budget_exhausted" if budget_exhausted else "no_valid_plan_found"
            ),
            optimality_claim=False,
            records=tuple(records),
            stats=stats,
        )

    def _priority(self, node: _SearchNode) -> float:
        """Return f(n) = g(n) + h(n), weighted for best-effort modes."""
        if self.mode == "astar":
            return node.g_cost + node.h_cost
        return node.g_cost + self.weight * node.h_cost

    def _beam_search(
        self,
        plan: PartialOrderPlan,
        goal: GoalSpec,
        operators: Mapping[str, OperatorSpec],
        initial_state: PlanState,
        quality_threshold: float,
    ) -> SearchOutcome:
        """Run level-synchronous beam search and record every pruned branch."""
        started = time.monotonic()
        records: list[SearchRecord] = []
        counters = {"generated": 0}
        seen: dict[str, float] = {}
        expanded = 0
        duplicates_total = 0
        dominance_total = 0
        beam_pruned = 0
        budget_exhausted = False
        best: _SearchNode | None = None
        width = max(1, self.budget.beam_width)

        initial_hash = initial_state.state_key()
        seen[initial_hash] = 0.0
        level: list[_SearchNode] = [
            _SearchNode(
                g_cost=0.0,
                facts=frozenset(initial_state.achieved_facts),
                executed=frozenset(),
                sequence=(),
                state_hash=initial_hash,
                parent_hash=None,
                h_cost=self.heuristic.estimate(
                    initial_state, goal, operators, self.cost_model,
                ),
            ),
        ]

        while level:
            if expanded >= self.budget.max_search_nodes:
                budget_exhausted = True
                break
            if time.monotonic() - started > self.budget.max_time_seconds:
                budget_exhausted = True
                break
            ranked = sorted(
                level, key=lambda item: (self._priority(item), item.g_cost, item.sequence),
            )
            kept, pruned = ranked[:width], ranked[width:]
            beam_pruned += len(pruned)
            next_level: list[_SearchNode] = []
            for entry in kept:
                if self._goal_reached(goal, entry.facts, quality_threshold):
                    if best is None or entry.g_cost < best.g_cost:
                        best = entry
                    continue
                if entry.g_cost > self.budget.max_plan_cost:
                    budget_exhausted = True
                    continue
                if len(entry.sequence) >= self.budget.max_depth:
                    continue
                expanded += 1
                records.append(SearchRecord(
                    state_hash=entry.state_hash,
                    g_cost=entry.g_cost,
                    h_cost=entry.h_cost,
                    parent=entry.parent_hash,
                    chosen_operator=entry.sequence[-1][1] if entry.sequence else None,
                    depth=len(entry.sequence),
                ))
                successors, duplicates, dominance = self._successors(
                    entry, plan, goal, operators, initial_state, seen, counters,
                )
                duplicates_total += duplicates
                dominance_total += dominance
                next_level.extend(successors)
            level = next_level

        stats = SearchStats(
            expanded=expanded,
            generated=counters["generated"],
            duplicates_pruned=duplicates_total,
            dominance_pruned=dominance_total,
            beam_pruned=beam_pruned,
            budget_exhausted=budget_exhausted,
        )
        if best is not None:
            return SearchOutcome(
                operator_sequence=best.sequence,
                total_cost=best.g_cost,
                goal_achieved=True,
                complete=True,
                stop_reason="verified_success_conditions",
                optimality_claim=False,
                records=tuple(records),
                stats=stats,
            )
        fallback = self._best_partial(level, initial_state)
        return SearchOutcome(
            operator_sequence=fallback[0],
            total_cost=fallback[1],
            goal_achieved=False,
            complete=False,
            stop_reason=(
                "search_budget_exhausted" if budget_exhausted else "no_valid_plan_found"
            ),
            optimality_claim=False,
            records=tuple(records),
            stats=stats,
        )

    @staticmethod
    def _best_partial(
        frontier: list[Any], initial_state: PlanState
    ) -> tuple[tuple[tuple[str, str], ...], float]:
        """Return the deepest frontier entry when no goal-reaching plan was found."""
        del initial_state
        if not frontier:
            return ((), 0.0)
        candidates = [
            item if isinstance(item, _SearchNode) else item[2] for item in frontier
        ]
        deepest = max(candidates, key=lambda item: (len(item.sequence), -item.g_cost))
        return deepest.sequence, deepest.g_cost


def _apply_effects(facts: frozenset[str], operator: OperatorSpec) -> frozenset[str]:
    """Apply an operator's positive and negative effects to a fact set."""
    removed = {
        effect.to_atom()[4:-1]
        for effect in operator.effects
        if not effect.polarity
    }
    added = {effect.to_atom() for effect in operator.effects if effect.polarity}
    return frozenset((facts - removed) | added)


__all__ = ["SearchEngine", "SearchOutcome", "SearchStats"]
