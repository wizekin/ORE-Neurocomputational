# Layered Data Infrastructure — Point 18 Architecture

Implementation of the ORE Integrated Algorithm "Point 18: Probabilistic Data
Structures, Streaming Algorithms, Storage and Data Integrity" (source
sections 1–10, 56 algorithmic steps in Phases A–J) for
ORE-Neurocomputational, integrated with USG semantic knowledge, the shared
workspace, the SQLite execution-log family and the CROS resource-management
layer.

This document maps every step of the algorithm to its implementation and
states the same scientific caution the source document states: the package
implements a layered data infrastructure beneath ORE's cognitive
architecture, not a new knowledge authority. Approximate structures may
accelerate or prioritize work; they must never silently become authorities
for semantic truth, verified membership, exact counts or evidence
correctness (source 1.2). USG keeps those responsibilities.

## Package layout

```
ore_data/
├── contracts.py            Phase A — Steps 1–6   data contracts, canonical
│                                                 identities, exactness
│                                                 classes, freshness and
│                                                 error budgets
├── ingestion.py            Phase B — Steps 7–12  validation, idempotency,
│                                                 outbox publication,
│                                                 storage routing
├── index_registry.py       Phase B — Step 12     rebuildable derived-index
│                                                 registry with watermarks
├── probabilistic/
│   ├── bloom.py            Phase C — Steps 13–19 membership acceleration,
│   │                                             counting filter, reconciliation
│   ├── count_min.py        Phase D — Steps 20–25 frequency sketches with
│   │                                             namespace separation, rotation
│   └── hyperloglog.py      Phase E — Steps 26–31 cardinality estimation with
│                                                 compatible merging
├── streaming/
│   ├── quantiles.py        Phase F — Steps 32–34 KLL quantile sketch
│   └── heavy_hitters.py    Phase F — Steps 35–36 Space-Saving summaries with
│                                                 exact validation support
├── similarity/
│   ├── minhash.py          Phase G — Steps 37–39, 42–43 signatures, exact
│   │                                             verification, provenance
│   ├── lsh.py              Phase G — Step 40     band bucketing and thresholds
│   └── simhash.py          Phase G — Step 41     fingerprints and Hamming
│                                                 candidate detection
├── storage/
│   ├── lsm_adapter.py      Phase H — Steps 44–49 WAL, memtables, SSTables,
│   │                                             compaction, backup/restore
│   └── content_addressed.py Phase I — Steps 50–53 immutable object store,
│                                                 manifest references
├── integrity/
│   └── merkle.py           Phase J — Steps 54–55 domain-separated Merkle
│                                                 trees, proofs, commitments
├── recovery/
│   └── erasure.py          Phase J — Step 56    Reed-Solomon archives over
│                                                 GF(2^8)
├── controller_adapter.py   Section 6 — Main Brain coordinator, CROS budgets,
│                                                 workspace publication, USG
│                                                 verification adapters
├── observability.py        Section 8.5 — estimate metadata, metrics, health
├── adapters.py            Section 5 — the 16 proposed interfaces and protocol
│                           conformance checks
├── pipeline.py             Source section 2 — end-to-end runtime assembly
├── registry.py             The 56-operation registry (Phases A–J)
└── __main__.py             CLI summary
```

## Core architectural decisions

1. **Approximation never becomes authority (source 1.2).** Every approximate
   result is a typed object (`MembershipHint`, `FrequencyEstimate`,
   `CardinalityEstimate`, `QuantileEstimate`, `HeavyHitterCandidate`,
   `SimilarityCandidate`) carrying algorithm, version, parameters, error
   bound, source watermark and timestamp (`estimate_metadata`); `exact` is
   `False` and cannot be set true by callers.

2. **Watermark-gated Bloom negatives (Step 17).** `membership_hint` returns
   `DEFINITELY_ABSENT` only when the filter is known complete for the
   authoritative version being queried; a stale or incomplete filter returns
   `UNKNOWN_INCOMPLETE` and the exact authoritative lookup runs anyway.
   Positives always proceed to exact resolution.

3. **No silent Bloom deletion (Step 18).** `BloomFilter.delete` deliberately
   raises; deletion goes through `rebuild()` or the separately validated
   `CountingBloomFilter` with underflow guards.

4. **Commit before derived updates (Steps 9–12).** `IngestionService` writes
   the authoritative store first, then publishes change events through an
   outbox; duplicate idempotency keys never re-apply mutations and never
   double-update sketches. Every derived structure is registered in
   `DerivedIndexRegistry` with source, schema version, build watermark,
   parameters, checksum and rebuild procedure, so it can always be discarded
   and rebuilt without losing authoritative knowledge.

5. **Candidates are nominated, never merged (Steps 40–43).** LSH buckets and
   SimHash distances only nominate; `verify_candidate_similarity` performs the
   exact set comparison, byte-identity check or structured field comparison
   before any merge, and `record_duplicate_provenance` preserves algorithm,
   thresholds, scores and verification results.

6. **Domain separation everywhere (Steps 3, 54).** Canonical serialization
   plus domain-tagged hashing gives every object class its own hash
   namespace: content identifiers, Merkle leaf hashes and internal node
   hashes cannot collide across types.

7. **LSM stays dormant until justified (Step 44).** The default deployment
   does not build an LSM store; `lsm_justified=True` activates the WAL /
   memtable / SSTable / compaction path with crash recovery, tombstone-safe
   compaction under CROS budgets, backup validated by restoration, and
   per-table Bloom filters.

8. **Graceful degradation (source 8.4).** Unavailable or stale analytics stay
   typed: an empty KLL sketch returns an estimate explicitly marked
   `available=False`; an incomplete derived index is bypassed for prechecks
   and reconciliation schedules a deterministic rebuild.

## USG and CROS integration

- `controller_adapter.USGSemanticVerificationAdapter` resolves candidate
  merges through an authoritative concept resolver (with an
  `UsgIndexResolver` that adapts the repository's own `ore_usg.retrieval`
  index interface); similarity alone never merges distinct USG concepts, and
  byte-identical duplicates remain the only identity-free merge path.
- `CROSDataBudgetAdapter` converts storage telemetry into bounded scheduling
  adaptation (compaction batches, rebuild intervals); correctness and
  integrity parameters sit outside the adaptable set and cannot be relaxed.
- `WorkspaceDataPublisher` emits versioned change events with sequence
  metadata and publishes derived-view watermarks; derived indexes are views,
  never workspace authority.
- `MainBrainCoordinatorAdapter` receives estimate metadata and the exactness
  class with every data-service answer, so the coordinator can select exact
  or approximate services explicitly.

## Acceptance coverage

The test suite (`tests/test_data_*_point18.py`, 93 tests) covers the
acceptance set of the source document section 8, including: workload
inventory before structure adoption; data-family contracts; deterministic
domain-separated canonicalization; invalid records quarantined without
index or commitment updates; idempotent retries that never duplicate
authoritative mutations; replay-safe versioned events; storage routing by
mutability class; Bloom sizing matching the m and k formulas; zero false
negatives with a bounded measured false-positive rate; watermark-gated
two-stage membership; refused standard-filter deletion with counting-filter
underflow guards; reconciliation with measured false-positive rates and
rebuild recommendations; prechecks that reduce exact lookups; Count-Min
dimensions matching the e/eps and ln(1/delta) formulas with validated
non-negative weights, namespace separation, rotation separating recent from
lifetime frequency; HLL accuracy across small/medium/large cardinalities
with small-range correction, compatible merges and incompatible rejection;
KLL quantile estimates tracking exact percentiles with seeded deterministic
compaction and bounded memory; Space-Saving heavy hitters bounded by W/k and
validated against exact aggregation; canonical shingles; MinHash Jaccard
estimates within tolerance; LSH candidate probability following the banding
formula with measured recall; SimHash fingerprints and Hamming candidates;
exact verification before any merge with byte-identity and structured-field
paths; duplicate classification with provenance; LSM WAL crash recovery and
corrupt-tail truncation; newest-first reads; tombstone semantics with safe
compaction; CROS-deferred compaction; per-table Bloom read skipping;
backup validated by restoration; adoption gated on measured benefit;
content-addressed writes that verify and deduplicate; rejection of mutable
artifacts without versioning; manifest-separated references; verified reads
that recompute hashes; Merkle roots with metadata, proofs that verify every
leaf, fail on any alteration, with domain-separated tags; Reed-Solomon
archives reconstructing from any sufficient shard set, rejecting corrupt
shards, requiring k valid shards, with standalone metadata round-trips and
GF(2^8) field arithmetic; coordinator estimates carrying metadata; CROS
bounds protecting correctness; bounded backpressure; workspace typed events
and derived views; USG gates on similarity merges; all 56 registry
operations in 10 phases; and the 16 proposed section 5 interfaces.

## Status and limits

- All sketch error bounds are the documented nominal guarantees under their
  standard assumptions; measured error rates depend on real ORE workloads
  and remain telemetry to collect, not claims already made (source 8.2).
- The LSM store is fully implemented but dormant by default; adoption
  requires a benchmark that demonstrates a write-heavy benefit over the
  existing SQLite path (Step 44).
- Erasure coding applies to selected archives only; it is not a replacement
  for backups, access control or cryptographic integrity (Step 56).
- The content-addressed store uses domain-separated SHA-256; a matching hash
  proves byte integrity relative to the identifier, never the safety or
  correctness of the artifact (source 1.2).
