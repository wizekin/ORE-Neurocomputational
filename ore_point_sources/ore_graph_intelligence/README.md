# ORE Point 8 — Graph Neural Networks & Neural Algorithmic Reasoning

**Package:** `ore_graph_intelligence` v1.0.0
**Implements:** "ORE —Integrated Algorithm Point 8_ Graph Neural Networks and Neural Algorithmic Reasoning.txt" (Algorithms_Chats branch)
**Target:** ORE-Neurocomputational framework
**Status:** fully implemented; 56 tests passing (`pytest ore_graph_intelligence/tests/`)

## What this package is

A bounded **USG Graph Intelligence Layer** for ORE-Neurocomputational. It projects
authoritative USG (Unified Semantic Graph) concepts and relations into typed graphs,
learns structural representations with knowledge-graph embeddings and relational GNNs,
proposes candidate relations, and sends every candidate through symbolic and
evidence-based verification before any authorized knowledge update — plus an isolated
**Neural Algorithmic Reasoning (NAR)** research track compared against deterministic
algorithms.

**Core invariant:** learned scores guide search; they do not establish truth. An
unobserved edge is not necessarily false. A high score is not a proof.

## Package contents

```
ore_graph_intelligence/
├── __init__.py            # public API, v1.0.0
├── schemas/               # typed graph, candidate, model-manifest schemas
│   ├── graph_schema.py        # RelationSignature, Cardinality, GraphSchema
│   ├── candidate_schema.py    # GraphCandidate, dispositions, namespaces
│   └── model_manifest.py      # ModelManifest, training provenance
├── projection/            # Phase A: read-only USG graph projection
│   ├── usg_snapshot.py        # USGGraphProjector, TypedGraphSnapshot
│   ├── typed_graph.py         # typed adjacency, graph partitioner
│   ├── provenance.py          # provenance, quarantine records
│   └── integrity.py           # IntegrityValidator, graph hash
├── embeddings/            # Phase B: KGE models + leakage-safe training
│   ├── model_adapters.py      # TransE, DistMult, ComplEx, RotatE
│   ├── trainer.py             # EmbeddingTrainer, calibration
│   └── splitting.py           # leakage-safe split policies & manifest
├── gnn/                   # Phase C: relational GNN stack
│   ├── rgcn.py                # R-GCN encoder (primary model)
│   ├── gcn.py / graphsage.py / gat.py   # measured alternatives
│   └── inference.py           # GNNInferenceEngine, baselines, oversmoothing monitor
├── discovery/             # Phase D: candidate discovery
│   ├── candidate_generator.py # anchor expansion, typed candidates
│   ├── link_predictor.py      # complementary scoring, normalization
│   ├── ranker.py               # novelty/utility ranking
│   ├── abstention.py           # abstention policy
│   └── explainer.py            # neighborhood explanations
├── verification/          # Phase D: symbolic + evidence verification
│   ├── type_checker.py         # relation-arity/type validation
│   ├── contradiction_search.py # counter-evidence search
│   ├── evidence_checker.py     # graded evidence classification
│   ├── usg_rule_adapter.py     # USG rule port
│   └── promotion_gate.py      # PromotionGate, AuthorizedUpdateProposal
├── nar/                   # Phase E (steps 41-48): isolated NAR research track
│   ├── task_generators.py      # narrow algorithmic tasks
│   ├── trace_dataset.py       # deterministic execution traces
│   ├── algorithmic_gnn.py     # stepwise/final supervision trainer
│   └── generalization_eval.py # OOD evaluation, production gate
├── observability/         # step 40: lifecycle events, metrics, audit
│   ├── events.py / metrics.py / audit.py
├── orchestration/         # Phase F: end-to-end service
│   ├── service.py              # GraphIntelligenceCoordinator
│   ├── workspace_bridge.py    # ORE workspace integration
│   ├── model_registry.py       # staged model deployment
│   ├── resource_estimator.py   # admission control
│   └── cache.py                # result cache
└── tests/                 # 56 tests, all passing
    ├── fixtures.py
    ├── test_projection_unit.py
    ├── test_embeddings_gnn_unit.py
    ├── test_discovery_verification_unit.py
    ├── test_nar_unit.py
    ├── test_orchestration_integration.py
    ├── test_e2e_integration.py
    ├── test_leakage.py / test_adversarial.py / test_benchmarks.py
```

## Dependencies

- **Runtime:** Python 3.9+ and `numpy` (pure NumPy/stdlib implementation; no PyTorch required)
- **Tests only:** `pytest>=7.4`
- **Framework:** this package is an ORE-Neurocomputational module and must sit in the
  repository root next to the other `ore_*` packages. It resolves the following sibling
  packages at import time (transitive closure, all already present in the repo):

  | direct | transitively required by those |
  |---|---|
  | `ore_usg` | `ore_contracts`, `ore_lexical_context`, `ore_semantic` |
  | `ore_workspace` | `ore_logger`, `ore_lexical_access`, `ore_syntax`, `ore_discourse`, `ore_semantic_composition`, `ore_n1`, `ore_n2`, `ore_n3`, `ore_n4`, `ore_n9` |

  Note: `ore_usg` and `ore_workspace` are imported lazily (inside functions), so the
  package imports cleanly even in isolation, but full tests require the whole framework.

## Integration into ORE-Neurocomputational

1. Copy the `ore_graph_intelligence/` directory into the repository root
   (next to the other `ore_*` packages).

2. Register it in `pyproject.toml`:

   - `[tool.setuptools.packages.find]` — add `"ore_graph_intelligence*"` to `include`
   - `[tool.pytest.ini_options]` — set `testpaths = ["tests", "ore_graph_intelligence/tests"]`

3. Verify the install:

   ```bash
   python -c "import ore_graph_intelligence as m; print(m.__version__)"
   # -> 1.0.0
   pytest ore_graph_intelligence/tests/ -q
   # -> 56 passed
   ```

## Data-flow summary

```
USG (authoritative)                                  ORE workspace
  │                                                        ▲
  ▼                                                        │ authorized,
projection ──► embeddings / gnn ──► discovery ──► verification ──┘ checked updates
  (typed, read-only,         (R-GCN primary,     (complementary      (type check,
   integrity-checked)         KGE baselines)     scoring, ranking,    contradiction,
                                                   abstention)         evidence, gate)
                                                  │
                                                  ▼
                                         observability (events, metrics, audit)
                                         orchestration (admission, cache, registry)

NAR research track (isolated): task generators → trace dataset → algorithmic GNN
→ OOD generalization eval → production gate. Advisory only; deterministic checks
before any production use.
```

## Safety boundaries

- The USG projection is **read-only**; the layer never writes to the USG directly.
- Candidates live in a **hypothesis namespace** (`schemas.new_hypothesis_namespace`),
  never in the authoritative namespace.
- Promotion to an authorized update proposal requires passing the full verification
  chain (type check → contradiction search → evidence check → promotion gate).
- NAR outputs are **advisory** and deterministically checked; the production gate
  blocks deployment unless generalization thresholds are met.
