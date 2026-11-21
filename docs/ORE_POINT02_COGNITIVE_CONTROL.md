# Point 02 — Cognitive Architectures: Soar, ACT-R and CLARION

## Scope of this implementation increment

This package establishes a small, testable cognitive-control foundation inspired by three architecture families:

- **Soar:** explicit operator applicability, deterministic selection, and a bounded decision cycle.
- **ACT-R:** declarative retrieval activation using recency/frequency, cue overlap, task relevance, provenance when available, and retrieval cost.
- **CLARION:** explicit separation of declarative retrieval and procedural operators behind typed contracts and a registry, allowing later addition of distinct implicit/explicit learning subsystems.

This is a foundation increment, not a claim that all 40 source-specification steps or the complete Soar, ACT-R, or CLARION architectures have been implemented.

## Modules

- `models.py`: immutable context, budgets, retrieval records, operator result, and trace contracts.
- `retrieval.py`: ACT-R-inspired base-level activation and deterministic candidate ranking.
- `registry.py`: thread-safe versioned operator registry.
- `engine.py`: one bounded selection/execution step, capability checks, preconditions, deterministic tie-breaking, and trace callback.
- `tests/test_cognitive_control_point02.py`: focused unit tests.

## Retrieval mathematics

For memory item (i), base-level activation is approximated by:

[
B_i = \ln\left(\sum_j (t_{now}-t_{ij})^{-d}\right)
]

where (t_{ij}) are access timestamps and (d) is the decay parameter. The implementation clamps non-positive ages and returns negative infinity for an item with no access history. The normalized activation is combined with cue overlap, task-tag overlap, optional provenance score, and an explicit estimated-cost penalty.

Missing provenance is not invented. Unavailable candidates are excluded.

## Decision cycle

1. Validate the context and budget contracts.
2. Enumerate registered operator versions deterministically.
3. Check required capabilities.
4. Evaluate operator preconditions.
5. Rank applicable operators by task relevance and estimated cost, with stable tie-breaking.
6. Execute at most one selected operator.
7. Emit a structured trace, including failures and elapsed time.

The engine does not run an internal loop, mutate a persistent ORE workspace, or silently commit the returned state patch. A caller must validate and commit that patch through the appropriate ORE workspace API.

## Deliberate integration boundary

The current repository exposes a rich `ore_workspace` package, but this increment does not guess signatures for its state manager, goal evaluator, event bus, retrieval stores, USG, logger, or task/resource coordinators. Adapters should be implemented only after each concrete contract has been inspected and covered by integration tests.

Deferred work includes learned operator acquisition, chunking, utility learning, subsymbolic parameter fitting, CLARION dual-representation learning, goal-stack persistence, impasse/subgoal resolution, conflict-driven deliberation, long-term memory storage, and asynchronous multi-agent scheduling.

## Test

From the repository root:

```bash
python -m pytest tests/test_cognitive_control_point02.py -q
```
