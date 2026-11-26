"""Source-ordered operation registry for the 56 steps of Point 18.

Maps every algorithmic step of the probabilistic data structures, streaming
algorithms, storage and data integrity specification to its implementation
module, phase, protocol rule and architectural reference, following the
repository's registry convention.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

#: Version of the Point 18 implementation.
ORE_DATA_VERSION = "0.1.0"

#: Contract identifier for the layered cognitive data infrastructure.
ORE_DATA_CONTRACT = "ORE.DATA.INFRASTRUCTURE/0.1.0"


@dataclass(frozen=True, slots=True)
class OperationSpec:
    """Describe one Point 18 operation and its implementation mapping."""

    operation_id: str
    phase: str
    source_step: int
    name: str
    purpose: str
    protocol_rule: str
    implementation_module: str
    evidence_level: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Return the operation record as a plain dictionary."""
        return asdict(self)


_ROWS: tuple[tuple[int, str, str, str, str, str, tuple[str, ...], tuple[str, ...]], ...] = (
    # Phase A - establish data contracts and authority boundaries
    (1, "A", "Inventory current ORE data stores and indexes", "Inspect USG storage, SQLite logger databases, workspace persistence, lexical and semantic indexes, artifact directories and remote storage before proposing new structures.", "no proposed structure is adopted before the current workload is measured", "ore_data.contracts", ("store_descriptors",), ("data_store_inventory",)),
    (2, "A", "Classify data by correctness and mutability", "Assign each data family a contract: exact transactional state, append-only event, immutable artifact, approximate telemetry, derived index or cache.", "a probabilistic index never replaces an exact source of record", "ore_data.contracts", ("data_families",), ("data_contracts",)),
    (3, "A", "Define canonical identities and serialization", "Deterministic keys and serialization for concepts, relations, documents, tasks, events, code artifacts and snapshots with domain separation.", "id(b) = H(domain_tag || b); byte-identical artifacts share one content identifier", "ore_data.contracts", ("records",), ("canonical_bytes", "content_ids")),
    (4, "A", "Define exactness classes", "Specify per operation whether it requires exact membership, bounded-error estimation, candidate similarity or cryptographic integrity.", "USG concept identity resolution requires exact authoritative lookup", "ore_data.contracts", ("operations",), ("exactness_classes",)),
    (5, "A", "Define freshness and consistency contracts", "Specify maximum acceptable lag between authoritative writes and each derived structure, and the update mode of every structure.", "stale filter or sketch exposes its version and freshness status", "ore_data.contracts", ("derived_structures",), ("freshness_contracts",)),
    (6, "A", "Establish resource and error budgets", "Define memory budget, expected cardinality, target error rate, update and query rate and rebuild cost per approximate structure as versioned configuration.", "CROS recommends allocation; the data service enforces correctness and memory limits", "ore_data.contracts", ("approximate_structures",), ("error_budgets",)),
    # Phase B - build the ingestion and update path
    (7, "B", "Validate each incoming record", "Check schema version, required fields, canonical identity, provenance, timestamps, authorization and size limits before any index update.", "invalid records are rejected or quarantined; they never update indexes or integrity commitments", "ore_data.ingestion", ("incoming_records", "provenance"), ("validation_results",)),
    (8, "B", "Assign a durable event identity", "Use a stable event ID or idempotency key so retries never double-apply updates; record type, entity key, source and causal reference.", "same idempotency key -> same authoritative mutation applied once", "ore_data.ingestion", ("validated_records",), ("event_identities",)),
    (9, "B", "Commit authoritative state before derived updates", "Write the authoritative record transactionally first; use an outbox when commit and publication cannot share one transaction.", "derived index updates are reported committed only after the authoritative source is durable", "ore_data.ingestion", ("validated_records",), ("commit_receipts",)),
    (10, "B", "Publish versioned data-change events", "Emit insert, update, delete, snapshot, merge and integrity events with sequence metadata; consumers detect duplicates and replay safely.", "events carry sequence/version metadata; replay is idempotent", "ore_data.ingestion", ("commit_receipts",), ("change_events", "event_receipts")),
    (11, "B", "Route records to the correct storage class", "Mutable task and workspace state stays transactional; immutable artifacts route to content-addressed storage; append-heavy records may use the LSM path.", "routing follows the data-family contract, not convenience", "ore_data.ingestion", ("commit_receipts", "contracts"), ("storage_routes",)),
    (12, "B", "Maintain a rebuildable derived-index registry", "Every Bloom filter, sketch, similarity index and materialized view declares source, schema version, build watermark, parameters, checksum and rebuild procedure.", "derived structures can be discarded and rebuilt without losing authoritative knowledge", "ore_data.index_registry", ("derived_structures",), ("index_registry", "rebuild_plans")),
    # Phase C - Bloom filter membership acceleration
    (13, "C", "Select Bloom filter use cases", "Use Bloom filters for high-volume membership prechecks such as lexical shard lookups or artifact-cache presence, never where false positives trigger unsafe behavior.", "avoid where exact indexes are fast enough or positives are dangerous", "ore_data.probabilistic.bloom", ("workload_measurements",), ("bloom_use_cases",)),
    (14, "C", "Size the bit array from expected cardinality and error", "Derive optimal bit count from expected insertions and target false-positive probability, rounding to a practical allocation.", "m = ceil(-n ln p / (ln 2)^2)", "ore_data.probabilistic.bloom", ("expected_keys", "target_fp_rate"), ("bloom_parameters",)),
    (15, "C", "Choose the number of hash probes", "Derive the optimal number of hash functions and use double hashing instead of k independent expensive hashes.", "k = round((m/n) ln 2); probes via double hashing", "ore_data.probabilistic.bloom", ("bloom_parameters",), ("probe_count",)),
    (16, "C", "Implement insertion", "Set each hash-selected bit with atomic-safe updates; persist the filter version and insertion watermark.", "bits set at (h1 + i*h2) mod m; watermark advances with inserts", "ore_data.probabilistic.bloom", ("keys",), ("set_bits", "insertion_watermark")),
    (17, "C", "Query membership with a safe two-stage path", "A negative may skip the exact lookup only when the filter is complete for the queried authoritative version; a positive always performs the exact lookup.", "definitely_absent iff complete watermark; possibly_present -> exact lookup", "ore_data.probabilistic.bloom", ("keys", "authoritative_watermark"), ("membership_hints",)),
    (18, "C", "Handle deletion and rebuilds correctly", "A standard Bloom filter never supports arbitrary deletion; use rebuilds, immutable generations or a separately validated counting filter.", "no delete on standard filter; counting filter guards counter underflow", "ore_data.probabilistic.bloom", ("deletion_requests",), ("rebuilds", "counting_filter_ops")),
    (19, "C", "Reconcile filter state against authoritative data", "Periodically sample exact records and absent keys, estimate the observed false-positive rate, verify watermarks and rebuild beyond policy.", "incomplete filter -> bypass the negative shortcut and query the authoritative index", "ore_data.probabilistic.bloom", ("authoritative_samples",), ("reconciliation_reports",)),
    # Phase D - frequency and usage sketches
    (20, "D", "Define Count-Min Sketch monitoring objectives", "Track approximate frequencies for concept retrieval, module activation, task patterns, error categories and repeated queries in separate namespaces.", "unrelated keys never contaminate one another: namespace separation", "ore_data.probabilistic.count_min", ("monitoring_objectives",), ("sketch_namespaces",)),
    (21, "D", "Size Count-Min Sketch dimensions", "Derive width and depth from the additive error and failure probability with documented guarantees.", "w = ceil(e/epsilon), d = ceil(ln(1/delta))", "ore_data.probabilistic.count_min", ("epsilon", "delta"), ("sketch_dimensions",)),
    (22, "D", "Update the sketch for each eligible event", "Increment one counter per row at the hash-selected column; support weighted updates only with validated non-negative weights; track total weight and epoch.", "weight >= 0 validated before update; epoch and window tracked", "ore_data.probabilistic.count_min", ("events", "weights"), ("sketch_counters", "total_weight")),
    (23, "D", "Estimate frequency", "Estimate with the minimum over rows and interpret as an upper-biased approximation, never as an exact occurrence count.", "f_hat(x) = min_i C[i, h_i(x)]; estimate >= true frequency", "ore_data.probabilistic.count_min", ("keys",), ("frequency_estimates",)),
    (24, "D", "Use frequency estimates for prioritization, not truth", "Prioritize caching, prefetching, index warming or diagnostics; exact counts require the authoritative event store.", "estimates drive prioritization only; exact counts query the source", "ore_data.probabilistic.count_min", ("frequency_estimates",), ("prioritization_decisions",)),
    (25, "D", "Control aging and time windows", "Rotate sketches by epoch buckets or documented decay so old activity does not dominate; keep lifetime and recent frequency distinct.", "rotation preserves lifetime vs recent frequency distinction", "ore_data.probabilistic.count_min", ("epochs",), ("rotated_sketches", "recent_estimates")),
    # Phase E - cardinality estimation with HyperLogLog
    (26, "E", "Identify where distinct-count estimates are useful", "Estimate unique concepts, task categories, artifact hashes, active modules or retrieval diversity where approximate cardinality suffices.", "HLL used only where exact set storage is too expensive", "ore_data.probabilistic.hyperloglog", ("distinct_count_needs",), ("cardinality_use_cases",)),
    (27, "E", "Configure precision and memory", "Choose the precision parameter p (m = 2^p registers) from target relative error, memory budget and expected cardinality.", "RSE ~ 1.04/sqrt(m); precision recorded with every result", "ore_data.probabilistic.hyperloglog", ("target_error", "memory_budget"), ("hll_parameters",)),
    (28, "E", "Hash and split each item", "Hash canonical item identity with a stable hash; one part selects the register, the remaining bits determine the rank statistic.", "same hash and canonicalization across compatible producers", "ore_data.probabilistic.hyperloglog", ("canonical_items",), ("register_indices", "ranks")),
    (29, "E", "Update the selected register", "Set the register to the maximum of its existing value and the new rank.", "M[j] = max(M[j], rho(w))", "ore_data.probabilistic.hyperloglog", ("hashed_items",), ("register_maxima",)),
    (30, "E", "Estimate cardinality and apply corrections", "Use the validated harmonic-mean estimator with small- and large-range corrections; record estimator version and precision.", "E = alpha_m m^2 / sum(2^-M[j]) with documented corrections", "ore_data.probabilistic.hyperloglog", ("registers",), ("cardinality_estimates",)),
    (31, "E", "Merge compatible sketches", "Merge only when hash algorithm, canonicalization, precision and register semantics match; reject incompatible sketches.", "merge = register-wise max; incompatible merge rejected", "ore_data.probabilistic.hyperloglog", ("sketch_pairs",), ("merged_sketches", "merge_rejections")),
    # Phase F - streaming quantiles and heavy hitters
    (32, "F", "Select a streaming quantile algorithm", "Use a validated quantile sketch (KLL) to monitor latency and resource-consumption distributions without storing every sample.", "KLL selected for its accuracy characteristics", "ore_data.streaming.quantiles", ("monitoring_objectives",), ("quantile_sketch_config",)),
    (33, "F", "Update quantile summaries", "Feed each validated observation with timestamp, unit and source; keep exact observations distinct from approximate summary values.", "approximate summary values never reported as exact observations", "ore_data.streaming.quantiles", ("observations",), ("quantile_sketch_state",)),
    (34, "F", "Query operational percentiles", "Estimate p50, p90, p95 or p99 for CROS capacity analysis and coordinator observability, never for correctness claims about an individual execution.", "estimates feed capacity analysis, not per-execution correctness", "ore_data.streaming.quantiles", ("percentile_queries",), ("quantile_estimates",)),
    (35, "F", "Detect heavy hitters", "Use Space-Saving summaries to identify frequent concepts, task keys, event types or failure categories with documented error guarantees.", "true frequency within [estimate - W/k, estimate] under documented assumptions", "ore_data.streaming.heavy_hitters", ("streams",), ("heavy_hitter_candidates",)),
    (36, "F", "Validate heavy-hitter candidates", "Validate against exact aggregation over a bounded retained window before high-impact decisions; approximate rankings stay qualified.", "exact validation before high-impact decisions", "ore_data.streaming.heavy_hitters", ("retained_window", "exact_aggregates"), ("heavy_hitter_validation",)),
    # Phase G - similarity and duplicate candidate detection
    (37, "G", "Define the representation being compared", "Specify whether similarity covers token sets, shingles, normalized text, graph neighborhoods, artifact bytes or semantic features.", "representation, language, tokenizer and normalization versions recorded", "ore_data.similarity.minhash", ("representations",), ("feature_schemas",)),
    (38, "G", "Construct canonical feature sets", "Normalize text and generate stable token shingles or semantic feature identifiers preserving schema versions for comparability.", "shingles derive from normalized tokens under a versioned schema", "ore_data.similarity.minhash", ("texts", "schemas"), ("feature_sets",)),
    (39, "G", "Generate MinHash signatures", "Construct r permutation hashes and store the per-set minima; matching fraction estimates Jaccard similarity.", "sig(A)_i = min h_i(x); J_hat = matched_components/r", "ore_data.similarity.minhash", ("feature_sets",), ("minhash_signatures",)),
    (40, "G", "Build LSH candidate buckets", "Partition signatures into bands, hash each band to buckets and tune bands/rows to the similarity threshold and candidate volume.", "P(candidate) = 1 - (1 - s^r)^b", "ore_data.similarity.lsh", ("signatures", "threshold_config"), ("candidate_buckets", "candidate_sets")),
    (41, "G", "Generate SimHash fingerprints where suitable", "Accumulate signed weighted feature contributions into bit dimensions and compare fingerprints with Hamming distance.", "bits by accumulator sign; distance validated on representative data", "ore_data.similarity.simhash", ("weighted_features",), ("simhash_fingerprints", "hamming_distances")),
    (42, "G", "Verify candidate pairs exactly or semantically", "Compute exact set similarity, canonical byte equality, structured field comparison or USG-aware semantic comparison for every candidate.", "never merge or delete records solely because a bucket or distance suggests similarity", "ore_data.similarity.minhash", ("candidate_pairs",), ("verified_similarity",)),
    (43, "G", "Preserve duplicate-detection provenance", "Record algorithm and version, feature schema, thresholds, candidate score, verification result and reviewer distinctions.", "byte-identical, near-duplicate, semantically related and logically equivalent stay distinct", "ore_data.similarity.minhash", ("verifications",), ("duplicate_provenance_records",)),
    # Phase H - durable storage and LSM organization
    (44, "H", "Measure whether an LSM-based store is justified", "Benchmark write amplification, random-write cost, read patterns, compaction overhead, growth and query latency before adopting an LSM store.", "LSM adopted only when benchmarks demonstrate benefit over current storage", "ore_data.storage.lsm_adapter", ("workload_benchmarks",), ("lsm_justification_reports",)),
    (45, "H", "Design the write path", "Implement the WAL, memtable, immutable SSTables and background compaction; acknowledge writes per the durability policy.", "writes acknowledged only per the configured durability policy", "ore_data.storage.lsm_adapter", ("writes",), ("wal_records", "memtable_state", "acknowledgements")),
    (46, "H", "Flush memtables into sorted tables", "Freeze the full memtable and flush sorted entries into an immutable SSTable while new writes continue in a fresh memtable.", "flush is atomic; WAL recovery semantics preserved", "ore_data.storage.lsm_adapter", ("full_memtables",), ("sstables",)),
    (47, "H", "Optimize reads across levels", "Search the newest memtable and SSTables in recency order with per-table indexes and Bloom filters to avoid unnecessary disk reads.", "duplicate keys resolved by version/tombstone semantics", "ore_data.storage.lsm_adapter", ("read_requests",), ("read_results", "skipped_tables")),
    (48, "H", "Perform compaction under resource budgets", "Merge sorted files, discard superseded versions when safe, preserve tombstones until values cannot reappear and monitor write amplification under CROS budgets.", "compaction never starves critical cognitive tasks", "ore_data.storage.lsm_adapter", ("sstables", "cros_budgets"), ("compaction_reports",)),
    (49, "H", "Define backup, restore and migration procedures", "Take consistent snapshots of manifests, SSTables and WAL state; test restore in a clean environment and verify counts, checksums and invariants.", "a backup is not validated until restoration has succeeded", "ore_data.storage.lsm_adapter", ("snapshot_requests",), ("backup_snapshots", "restore_reports")),
    # Phase I - content-addressed storage and immutable artifacts
    (50, "I", "Identify immutable ORE artifacts", "Route validated AlgoIR specifications, accepted algorithms, execution bundles, knowledge snapshots, model artifacts and archived reports to content-addressed storage.", "mutable workspace records are never forced into immutable storage without a versioning model", "ore_data.storage.content_addressed", ("artifact_classes",), ("immutable_candidates",)),
    (51, "I", "Compute content identifiers and store objects", "Serialize canonically, compute the domain-tagged cryptographic hash, store under that identifier, verify on write and read, and reuse existing objects.", "same content reuses the object instead of duplicating it", "ore_data.storage.content_addressed", ("canonical_bytes",), ("content_ids", "object_files")),
    (52, "I", "Separate object identity from mutable references", "Maintain a manifest mapping logical names and versions to immutable content IDs; updates create new objects and manifest entries.", "historical bytes are never overwritten", "ore_data.storage.content_addressed", ("logical_references",), ("manifest_entries",)),
    (53, "I", "Enforce safe object retrieval", "Resolve the content ID, retrieve bytes, recompute the hash, verify schema and provenance, then hand the artifact to the consumer.", "a matching hash proves byte integrity relative to the identifier, not safety or correctness", "ore_data.storage.content_addressed", ("content_ids",), ("verified_artifacts",)),
    # Phase J - Merkle integrity and erasure recovery
    (54, "J", "Build Merkle trees for selected snapshots and histories", "Hash canonical leaf records and pairs of child hashes with a defined odd-node rule; store the root with snapshot identity and metadata.", "leaf and node tags domain-separated; root stored with schema version and timestamp", "ore_data.integrity.merkle", ("canonical_records",), ("merkle_commitments",)),
    (55, "J", "Verify individual records using Merkle proofs", "Recompute the root from sibling hashes and path information; accept only when it equals the trusted committed root.", "consistency with the committed root, not factual correctness of the record", "ore_data.integrity.merkle", ("records", "proofs", "trusted_roots"), ("integrity_results",)),
    (56, "J", "Add erasure coding only for selected archives", "Evaluate Reed-Solomon or fountain codes for large archives; define shard counts, failure assumptions, reconstruction threshold, integrity checks and repair procedures.", "any k of n shards reconstruct; erasure coding never replaces backups or access control", "ore_data.recovery.erasure", ("archives", "shard_policies"), ("encoded_shards", "reconstruction_reports")),
)


def _build_registry() -> tuple[OperationSpec, ...]:
    """Build the immutable 56-operation registry."""
    specs = []
    for (
        step,
        phase,
        name,
        purpose,
        rule,
        module,
        inputs,
        outputs,
    ) in _ROWS:
        specs.append(
            OperationSpec(
                operation_id=f"ore.data.p18.step{step:02d}",
                phase=f"Phase {phase}",
                source_step=step,
                name=name,
                purpose=purpose,
                protocol_rule=rule,
                implementation_module=module,
                evidence_level="mathematical_guarantee_or_behavioral_test; acceptance set from source section 8",
                inputs=inputs,
                outputs=outputs,
            )
        )
    return tuple(specs)


#: Immutable registry of all 56 Point 18 operations.
OPERATION_REGISTRY: tuple[OperationSpec, ...] = _build_registry()


def operation_count() -> int:
    """Return the number of registered Point 18 operations."""
    return len(OPERATION_REGISTRY)


def phases() -> tuple[str, ...]:
    """Return the ordered phase labels of the registry."""
    ordered: list[str] = []
    for spec in OPERATION_REGISTRY:
        if spec.phase not in ordered:
            ordered.append(spec.phase)
    return tuple(ordered)


def registry_as_dicts() -> list[dict[str, object]]:
    """Return every operation record as a plain dictionary."""
    return [spec.to_dict() for spec in OPERATION_REGISTRY]


__all__ = [
    "ORE_DATA_VERSION",
    "ORE_DATA_CONTRACT",
    "OperationSpec",
    "OPERATION_REGISTRY",
    "operation_count",
    "phases",
    "registry_as_dicts",
]
