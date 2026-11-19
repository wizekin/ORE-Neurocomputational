# ORE Shared Cognitive Workspace (Task 2)

The `ore_workspace` package provides an independent, versioned communication layer for ORE modules. It is a shared state and controlled publication mechanism, not a central reasoning module. Producers submit findings; the state manager validates and commits them; subscribers receive events about committed changes.

## Lifecycle

1. `WorkspaceManager.create_workspace(input, goal)` creates a workspace and revision zero.
2. A module reads a detached `WorkspaceSnapshot` and computes independently.
3. `FindingPublisher` creates a typed finding with target, producer, evidence provenance, dependencies, confidence (optional), and base revision.
4. `WorkspaceStateManager` validates the schema, permission, revision, provenance and dependencies. Missing dependencies are deferred; malformed or unauthorized writes are rejected.
5. Field-specific merge policies add compatible information without overwriting competing hypotheses. Contradictory candidate labels create a `ConflictRecord` while preserving both findings.
6. The state manager commits atomically, stores a reconstructable revision, and publishes a bounded event to eligible subscribers. Subscriber failures are recorded individually and do not block other consumers.
7. `ModuleScheduler` supports weighted activation, asynchronous or synchronous handlers, timeouts, and bounded processing rounds.
8. `GoalEvaluator` combines weighted coverage with explicit critical requirements. A high aggregate score cannot hide a missing required representation.

## Mathematical operations

- **Subscription matrix:** \(A_{ij}=1\) when module \(i\) subscribes to field \(j\), otherwise zero.
- **Activation relevance:** \(R_m=\sum_j a_{mj}\Delta_j\); activation occurs when \(R_m\geq	heta_m\). Field sensitivity weights multiply change magnitudes.
- **Hypothesis evidence:** \(S(h)=\sum_i w_i e_i(h)\) combines source-weighted support. Contributions and provenance remain inspectable; correlated source groups are conservatively prevented from being double-counted. The result is an evidence score, not a calibrated probability.
- **Working-memory retention:** \(P_i=(lpha G_i+eta R_i+\gamma C_i+\delta U_i)/(lpha+eta+\gamma+\delta)\), with configurable non-negative weights and normalized factors. This is a selectable priority heuristic, not a claim about a literal neural equation.
- **Goal coverage:** \(C_{goal}=\sum_k w_k C_k/\sum_k w_k\). Explicit requirements are checked separately from the score.
- **Provenance and dependencies:** findings form a directed dependency graph; a finding with missing prerequisites waits rather than silently entering accepted state.
- **Idempotence:** content fingerprints prevent repeat submissions; per-field identity policies deduplicate memories and preserve hypothesis evidence.
- **Optimistic concurrency:** append-only hypotheses can be safely rebased against a newer revision; stale mutable-field updates are deferred for recomputation.

Scores are operational heuristics. Confidence fields are not treated as calibrated probabilities, and no biological validity is inferred from software tests.

## Logging

`register_workspace_annotations()` calls the existing `ore_logger.catalog.sync_repository` for `ore_workspace` only. Every module, class, function, and method is indexed in the existing `ore_logger.annotations` SQLite table. `SQLiteWorkspaceObserver` links runtime operations to those annotation IDs in the existing logger's `runs` and `events` tables. The logger observes execution; it is not the shared workspace or module message bus.

## Basic usage

```python
from ore_workspace import FindingPublisher, WorkspaceManager

with WorkspaceManager() as manager:
    workspace = manager.create_workspace("The boy opened the door.")
    lexical = FindingPublisher(workspace, "lexical_processor")
    result = lexical.publish(
        "lexical_hypotheses", {"candidate": "PERSON", "word": "boy"}, key="boy"
    )
    print(result.decision, workspace.snapshot().revision)
```

The default permission map follows module responsibilities in the Task 2 specification. Applications can provide explicit permission and subscription policies to `SharedWorkspace` / `WorkspaceManager`. Modules never receive unrestricted access to mutate one another's findings.