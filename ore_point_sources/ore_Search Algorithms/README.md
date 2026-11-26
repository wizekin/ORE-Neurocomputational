# ORE Point 12 — Search Algorithms and Hierarchical Planning (ore_planning)

This archive contains only the Point 12 planner implementation files. It is not
the whole repository. Extract it into the root of `wizekin/ORE-Neurocomputational`
so the paths line up with the existing package layout.

## Contents

| Path | Purpose |
| --- | --- |
| `ore_planning/contracts.py` | Typed, versioned data contracts: GoalSpec, OperatorSpec, HTNMethod, PlanNode, PlanEdge, PlanState, SearchRecord, PlanDecision, ExecutionFeedback, PlannerBudget, PlanCheckpoint, PlanRunResult |
| `ore_planning/goal_adapter.py` | Goal/constraint adapter: natural-language goals to typed predicates, ambiguity preservation, hard constraints, forbidden operations, USG/HIC interpreter hook |
| `ore_planning/capability_adapter.py` | Capability registry adapter: operator discovery from the existing ModuleRegistry, schema/capability/permission/dependency eligibility, transitive producer closure |
| `ore_planning/htn.py` | HTN decomposer: versioned methods, recursive expansion, cycle/depth/node guards, alternative preservation with provenance, default method library |
| `ore_planning/partial_order.py` | Partial-order planning: causal dependency DAG, concurrency levels, mutex/resource serialization, protection intervals, ready frontier, cycle detection |
| `ore_planning/cost.py` | Multi-dimensional cost model c(o) = w_t*T + w_m*M + w_r*R + w_u*U + w_f*F, goal progress P(G) with critical hard gates, calibration, admissible/inadmissible heuristics |
| `ore_planning/search.py` | A*, weighted A*, and bounded beam search with state hashing, duplicate/dominance pruning, node/time/cost budgets, and documented optimality claims |
| `ore_planning/executor.py` | Plan executor adapter: submits CognitiveTasks through the existing CognitiveTaskCoordinator, checkpoints, carry-forward resume |
| `ore_planning/replanner.py` | Outcome evaluator: prediction errors, revision-validated preconditions, replanning threshold E[dQ + dC_saved] > C_replan + C_disruption, stop conditions, explicit partial status |
| `ore_planning/provenance.py` | The 40-step specification trace registry and correlated provenance records (goal, plan, task, module, workspace revision) |
| `ore_planning/planner.py` | HierarchicalPlanner facade orchestrating phases A-E (all 40 specification steps) |
| `tests/test_planning_point12.py` | 32 acceptance tests covering every criterion in specification section 8 |

## Installation into the repository

1. Copy `ore_planning/` into the repository root and `tests/test_planning_point12.py`
   into `tests/`.
2. Register the package in `pyproject.toml` by adding `"ore_planning*"` to the
   `[tool.setuptools.packages.find] include` list (one line, next to the other
   `ore_*` entries).
3. Run the acceptance tests:

```bash
python -m pytest tests/test_planning_point12.py -q
```

## Usage

```python
from ore_contracts import ContractCatalog, ModuleRegistry
from ore_planning import HierarchicalPlanner, ZeroHeuristic, PlannerBudget

planner = HierarchicalPlanner(
    registry, catalog,
    budget=PlannerBudget(),
    heuristic=ZeroHeuristic(),
)
planning = planner.plan(
    "produce the semantic meaning of the input",
    workspace_revision=0,
    external_contracts=("ORE.TEST.INPUT",),
)
run = planner.execute(planning, input_artifacts={"source": artifact},
                      run_context=context)
```

## Boundaries (no duplicate subsystems)

The planner only creates and evaluates plans. Task execution stays with
CognitiveTaskCoordinator, scheduling with ModuleScheduler and the coordinator's
dependency planner, resource allocation with AdaptiveResourceCoordinator, and
authoritative state with ore_workspace. It registers no second scheduler,
workspace, goal store, or task coordinator.

## Scope notes

MCTS is intentionally not implemented: specification Stage 8 gates it behind
justified benchmarks with a measurable rollout model, which do not exist yet.
The UCT formula and its risks are documented in the specification; adding it
later requires only a new RemainingCostHeuristic-compatible search mode.

## Verification

- All 32 Point 12 acceptance tests pass.
- The full existing ORE regression suite remains green with the package added.
- ruff passes clean on the new module and tests.
