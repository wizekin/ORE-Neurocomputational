# Phase 5 — USG coordination and reasoning

Phase 5 connects the Phase 4 read-only knowledge interface to the Main Brain Coordinator (`ore_blackboard.BlackboardCoordinator`). It does not create a second knowledge store and does not mutate the source USG bundle.

## 16. Demand-driven knowledge routing

The coordinator asks each specialist manifest/capability which modules will participate in a task, closes the selected set over required specialist dependencies, and then considers the specialists' explicit `usg_knowledge_domains` declaration. The current reference specialists declare:

| Specialist | Routed USG domains |
| --- | --- |
| `ore.blackboard.lexical_specialist` | lexical, morphology, context, evidence |
| `ore.blackboard.semantic_specialist` | semantic, relations, predicates, context, evidence, source provenance |

If no selected specialist declares a USG domain, the coordinator does not create a `BlackboardUSGSnapshot`, does not hash/convert the supplied source bundle, and does not attach a USG artifact to the task. If a USG-using specialist is selected but no bundle is supplied, it continues with the available non-USG/fallback path.

A supplied bundle is digest-verified and converted once. `BlackboardCoordinator` retains a single bounded snapshot cache keyed by the canonical bundle digest. Subsequent tasks using the identical digest reuse that immutable snapshot and its lookup caches. A different digest replaces the one-entry cache. The expected digest is checked on cached reuse as well as on first load.

Each specialist invocation receives a dynamically built `ORE.BLACKBOARD.USG_MODULE_KNOWLEDGE@1.0.0` artifact. The slice includes only exact lexical entries requested for that task, linked concepts, directly incident explicit relations/memberships, matching predicate frames, and the explicitly attached evidence/context/source metadata required by that specialist. The slice artifact records the full knowledge-snapshot artifact hash as its parent; the full resource is not sent as a specialist input.

The router lazily builds snapshot-scoped exact indexes for lexical forms, semantic labels/aliases, predicate-frame keys, and incident relations/memberships. Each index is built at most once for that cached snapshot. Query-result caches for lexical terms, predicate terms, related concepts and completed module routes are each bounded to 2,048 entries and evict their oldest entry when full. The workspace `ComputationRegistry` continues to suppress repeated specialist computations when the task-relevant inputs are unchanged.

## 17. Evidence-aware semantic interpretation

The lexical specialist preserves every exact USG meaning returned for a surface form. It carries explicit semantic concept IDs/link statuses, morphological metadata, attached contexts, evidence statuses and source references into lexical findings.

The semantic specialist evaluates bounded combinations of candidate subject, predicate and object meanings. It gathers evidence from the candidate meanings and from explicit one-hop relations/category memberships attached to their concepts. Evidence is preserved as `SUPPORTS`, `CONTRADICTS`, `NEUTRAL` or `UNASSESSED`; the reference reasoning layer assigns signed contributions of +1, -1 or 0 for its inspectable comparison score.

Where both supporting and contradicting evidence occur, the interpretation is marked `CONFLICTED`. A purely contradicting set is marked `CONTRADICTED`. Competing meanings remain separate propositions; they are not merged and the coordinator does not force one winner. Predicate frames are attached to the matching predicate even when the frame is incomplete; incomplete frames remain non-executable and can generate a feedback request.

These scores are engineering heuristics, not calibrated probabilities. Evidence status is not proof, and no imported evidence is promoted to a verified fact.

## 18. Context and semantic feedback loop

Downstream specialists may put `usg_feedback_requests` in their finding payloads. Each request names a target module and carries a query or record ID, a feedback kind, and an explanation. The coordinator records the request in a detached `USGKnowledgeFeedback` ledger keyed by the bundle digest and request content.

Feedback is deduplicated, scoped for delivery once per module per task, and can be delivered again in a later task using the same cached bundle. The requested target module receives the feedback with its next domain-scoped knowledge slice. The coordinator also publishes each new request into the current ORE workspace's `unresolved_questions` field, allowing subscribers to run again. Identical feedback is not republished twice within the same workspace. An authorized ORE-side caller can close a feedback item through `router.resolve_feedback(feedback_id, status="RESOLVED")` or mark it obsolete with `status="SUPERSEDED"`; neither operation modifies the USG bundle.

Feedback is an ORE coordination record, not a source assertion. It does not modify `USGKnowledgeResource`, the detached bundle snapshot, evidence statuses, source records, or the USG database. A new bundle digest creates a new knowledge snapshot while old feedback remains tied to its original digest.

## 19. Resource coordination and work avoidance

`BlackboardResourceCoordinator.prioritize_knowledge_routes()` scores the demanded specialist routes with the existing resource estimator, reliability posterior, circuit-breaker status, latency/memory estimates and configured comparison coefficients. The returned `resource_plan.usg_knowledge_routes` exposes the route ordering and the estimates used. Its scores are explicitly heuristic; mandatory routes remain in the plan and are never silently skipped because an optional route scores higher.

Relevant work is further bounded by the declared domains and exact terms passed to `BlackboardUSGRouter`. Same-digest snapshot reuse, per-term/frame/related-concept caches, feedback deduplication and the existing computation registry avoid unnecessary bundle conversion, repeated lookups and repeated module execution. The coordinator trace and result resource plan retain route cache and feedback statistics.

## Local tests

Phase 5 adds `tests/test_usg_coordinator_integration.py`. Tests are added for local execution and were not run as part of this implementation.

```powershell
python -m pytest tests/test_usg_coordinator_integration.py -q
```

Run the full repository suite locally when you are ready:

```powershell
python -m pytest -q
```

