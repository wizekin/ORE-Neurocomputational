# ORE Point 18 — Probabilistic Data Structures, Streaming Algorithms, Storage and Data Integrity

Full implementation of the ORE Integrated Algorithm "Point 18: Probabilistic
Data Structures, Streaming Algorithms, Storage and Data Integrity" (56
algorithmic steps in Phases A–J) as the `ore_data` package for
ORE-Neurocomputational, integrated with USG semantic knowledge (`ore_usg`),
the shared cognitive workspace (`ore_workspace`), the execution-log family and
the CROS resource-management layer (`ore_contracts`).

The central engineering rule (source 1.2 / section 10): use the least
expensive structure that satisfies the operation's correctness contract.
Approximation belongs in filtering, estimation and candidate generation;
authoritative USG knowledge, verified conclusions and exact execution history
retain exact validation and provenance.

## Contents of this archive

```
ore_data/                             The algorithm package (28 modules)
├── contracts.py        Phase A  data contracts, canonical identities,
│                                exactness classes, freshness/error budgets (Steps 1–6)
├── ingestion.py        Phase B  validation, idempotency, outbox,
│                                authoritative commit, storage routing (Steps 7–11)
├── index_registry.py   Phase B  rebuildable derived-index registry (Step 12)
├── probabilistic/
│   ├── bloom.py        Phase C  watermark-gated membership acceleration,
│   │                             counting filter, reconciliation (Steps 13–19)
│   ├── count_min.py    Phase D  frequency sketches, namespaces, rotation (Steps 20–25)
│   └── hyperloglog.py  Phase E  cardinality estimation, compatible merges (Steps 26–31)
├── streaming/
│   ├── quantiles.py    Phase F  KLL quantile sketch (Steps 32–34)
│   └── heavy_hitters.py Phase F  Space-Saving summaries, exact validation (Steps 35–36)
├── similarity/
│   ├── minhash.py      Phase G  signatures, exact verification, provenance
│   │                             (Steps 37–39, 42–43)
│   ├── lsh.py          Phase G  band bucketing and thresholds (Step 40)
│   └── simhash.py      Phase G  fingerprints and Hamming candidates (Step 41)
├── storage/
│   ├── lsm_adapter.py  Phase H  WAL, memtables, SSTables, compaction,
│   │                             backup/restore, dormancy gate (Steps 44–49)
│   └── content_addressed.py Phase I  immutable objects, manifests (Steps 50–53)
├── integrity/
│   └── merkle.py       Phase J  domain-separated trees, proofs (Steps 54–55)
├── recovery/
│   └── erasure.py      Phase J  Reed-Solomon archives over GF(2^8) (Step 56)
├── controller_adapter.py  Section 6  coordinator, CROS budgets, workspace
│                                    publication, USG verification adapters
├── observability.py    Section 8.5  estimate metadata, metrics, index health
├── adapters.py         Section 5  the 16 proposed interfaces + conformance
├── pipeline.py         Source section 2  end-to-end runtime assembly
├── registry.py         The 56-operation registry (Phases A–J)
└── __main__.py         CLI summary (`python -m ore_data`)

tests/test_data_contracts_ingestion_point18.py    Steps 1–12 acceptance tests
tests/test_data_probabilistic_point18.py          Steps 13–31 acceptance tests
tests/test_data_streaming_point18.py              Steps 32–36 acceptance tests
tests/test_data_similarity_point18.py             Steps 37–43 acceptance tests
tests/test_data_storage_integrity_point18.py      Steps 44–56 acceptance tests
tests/test_data_integration_point18.py            Source sections 2, 5, 6, 8 acceptance tests

docs/DATA_INFRASTRUCTURE_POINT18_ARCHITECTURE.md  Step-to-code architecture map
examples/data_infrastructure.py                  End-to-end demonstration
```

## Installing into ORE-Neurocomputational

1. Copy `ore_data/` into the repository root (next to `ore_usg/`,
   `ore_workspace/`, `ore_contracts/`).
2. Copy the six `tests/test_data_*_point18.py` files into `tests/`.
3. Copy `docs/DATA_INFRASTRUCTURE_POINT18_ARCHITECTURE.md` into `docs/`.
4. Copy `examples/data_infrastructure.py` into `examples/`.
5. In `pyproject.toml`, append `"ore_data*"` to the
   `[tool.setuptools.packages.find] include` list (already done in the
   repository this archive was built against).

The package is dependency-light: it uses only the Python standard library.
The USG integration adapter optionally imports `ore_usg.retrieval` behind a
protocol interface; the core data structures never require it.

## Running

```bash
python -m pytest tests/test_data_*_point18.py -q     # 93 acceptance tests
python -m ore_data                                   # deployment + registry summary
PYTHONPATH=. python examples/data_infrastructure.py # end-to-end demonstration
```

## Scope notes

- Every approximate result is a typed object carrying algorithm, version,
  parameters, error bound, source watermark and timestamp (source 5.1);
  `exact` is permanently `False` for estimates.
- Bloom negatives only skip exact lookups when the filter's watermark proves
  completeness for the queried authoritative version; positives always
  proceed to exact resolution (Step 17). Standard filters refuse deletion;
  deletion uses rebuilds or the counting filter (Step 18).
- MinHash/LSH/SimHash nominate candidates only; exact set, byte-identity or
  structured verification plus USG identity resolution gate every merge
  (Steps 42–43, section 6).
- The LSM store is fully implemented but dormant by default: adoption
  requires a benchmarked write-heavy benefit over the existing SQLite path
  (Step 44).
- Merkle consistency proves byte integrity relative to a committed root; it
  is not factual correctness of the record (source 1.2). Erasure coding is
  for selected archives only and never replaces backups (Step 56).
