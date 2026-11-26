"""End-to-end data infrastructure pipeline (source section 2 as a runtime assembly).

The pipeline validates each incoming record, assigns its canonical identity,
commits authoritative state before derived updates, publishes versioned
change events, routes immutable artifacts to content-addressed storage,
updates exact and approximate side indexes, feeds the streaming sketches,
answers membership through the safe two-stage path, generates similarity
candidates with mandatory verification, builds Merkle commitments for
selected snapshots, reconciles derived indexes, and exposes metrics, error
bounds, freshness and integrity status to the coordinator. Approximate
structures only accelerate work here; they never become authorities.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from ore_data.adapters import verify_protocol_implementations
from ore_data.contracts import (
    DataClass,
    ExactnessClass,
    canonical_key,
    canonicalize,
    content_id,
    default_data_contracts,
)
from ore_data.controller_adapter import (
    BackpressureController,
    CROSDataBudgetAdapter,
    MainBrainCoordinatorAdapter,
    StorageTelemetry,
    WorkspaceDataPublisher,
)
from ore_data.index_registry import (
    DerivedIndexRegistry,
    DerivedIndexSpec,
    IndexKind,
    ReconciliationReport,
    reconcile_derived_indexes,
)
from ore_data.ingestion import (
    ChangeEventType,
    CommitReceipt,
    DataChangeEvent,
    IngestionService,
    ProvenanceInfo,
    RecordStatus,
    StorageRouter,
    ValidationPolicy,
)
from ore_data.integrity.merkle import (
    IntegrityVerifier,
    MerkleCommitment,
    MerkleTree,
    build_merkle_snapshot,
)
from ore_data.observability import (
    DataInfrastructureMetrics,
    IndexHealthMonitor,
    estimate_metadata,
)
from ore_data.probabilistic.bloom import (
    BloomFilter,
    BloomParameters,
    BloomPrecheckRouter,
    MembershipVerdict,
    reconcile_bloom_filter,
)
from ore_data.probabilistic.count_min import (
    CountMinParameters,
    CountMinSketch,
    FrequencyEstimate,
)
from ore_data.probabilistic.hyperloglog import (
    CardinalityEstimate,
    HyperLogLog,
    required_precision_for,
)
from ore_data.recovery.erasure import (
    ErasureArchive,
    ErasurePolicy,
)
from ore_data.registry import (
    ORE_DATA_CONTRACT,
    ORE_DATA_VERSION,
    OPERATION_REGISTRY,
    operation_count,
    phases,
)
from ore_data.similarity.lsh import LshBandConfig, LshBandIndex, choose_bands
from ore_data.similarity.minhash import (
    DEFAULT_PERMUTATIONS,
    DuplicateClassification,
    FeatureRepresentation,
    FeatureSchema,
    MinHashSigner,
    MinHashSignature,
    VerifiedSimilarity,
    build_shingles,
    classify_duplicate,
    estimated_jaccard,
    exact_jaccard,
    record_duplicate_provenance,
    verify_candidate_similarity,
)
from ore_data.similarity.simhash import SimHashFingerprint, SimHashIndex, simhash
from ore_data.storage.content_addressed import (
    ContentAddressedStore,
    StoreReceipt,
    VerifiedArtifact,
)
from ore_data.storage.lsm_adapter import LsmStore
from ore_data.streaming.heavy_hitters import (
    HeavyHitterCandidate,
    SpaceSavingSummary,
)
from ore_data.streaming.quantiles import KLLQuantileSketch, QuantileEstimate


@dataclass(slots=True)
class PipelineConfig:
    """Configuration of one data-infrastructure deployment (Steps 5-6, 13-15, 21, 27)."""

    workspace_root: Path | None = None
    expected_membership_keys: int = 4096
    bloom_target_fp_rate: float = 0.01
    count_min_epsilon: float = 0.005
    count_min_delta: float = 0.01
    hll_precision: int = 12
    kll_k: int = 128
    heavy_hitter_capacity: int = 64
    minhash_permutations: int = 64
    similarity_threshold: float = 0.8
    lsm_justified: bool = False
    erasure_enabled: bool = False
    sketch_rotation_interval_events: int = 10_000
    bloom_rebuild_interval_events: int = 1_000
    authorized_sources: frozenset[str] | None = None

    def __post_init__(self) -> None:
        """Validate the deployment configuration."""
        if self.expected_membership_keys <= 0:
            raise ValueError("expected_membership_keys must be positive")
        if not 0.0 < self.bloom_target_fp_rate < 1.0:
            raise ValueError("bloom_target_fp_rate must be within (0, 1)")
        if self.count_min_epsilon <= 0 or not 0.0 < self.count_min_delta < 1.0:
            raise ValueError("count-min epsilon/delta out of range")
        if not 4 <= self.hll_precision <= 16:
            raise ValueError("hll_precision must be within [4, 16]")
        if self.kll_k < 8:
            raise ValueError("kll_k must be at least 8")
        if self.heavy_hitter_capacity <= 0:
            raise ValueError("heavy_hitter_capacity must be positive")
        if self.minhash_permutations < 2:
            raise ValueError("minhash_permutations must be at least 2")
        if not 0.0 < self.similarity_threshold < 1.0:
            raise ValueError("similarity_threshold must be within (0, 1)")


@dataclass(slots=True)
class DataInfrastructureCapabilities:
    """Which data-infrastructure subsystems one deployment instantiated."""

    bloom_active: bool
    count_min_active: bool
    hyperloglog_active: bool
    quantile_active: bool
    heavy_hitter_active: bool
    similarity_active: bool
    content_addressed_active: bool
    merkle_active: bool
    lsm_active: bool
    erasure_active: bool

    def to_dict(self) -> dict[str, object]:
        """Return the capability set as a plain dictionary."""
        return {
            "bloom_active": self.bloom_active,
            "count_min_active": self.count_min_active,
            "hyperloglog_active": self.hyperloglog_active,
            "quantile_active": self.quantile_active,
            "heavy_hitter_active": self.heavy_hitter_active,
            "similarity_active": self.similarity_active,
            "content_addressed_active": self.content_addressed_active,
            "merkle_active": self.merkle_active,
            "lsm_active": self.lsm_active,
            "erasure_active": self.erasure_active,
        }


@dataclass(slots=True)
class ProcessingOutcome:
    """Outcome of one record processed end to end (source section 2)."""

    status: str
    entity_key: str | None = None
    record_id: str | None = None
    commit_receipt: CommitReceipt | None = None
    event_receipts: list[dict[str, object]] = field(default_factory=list)
    content_id: str | None = None
    content_reused: bool = False
    bloom_watermark: int = 0
    similarity_candidates: list[str] = field(default_factory=list)
    verification_required: bool = False
    errors: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        """Return the outcome as a plain dictionary."""
        return {
            "status": self.status,
            "entity_key": self.entity_key,
            "record_id": self.record_id,
            "commit_receipt": self.commit_receipt.to_dict() if self.commit_receipt else None,
            "event_receipts": list(self.event_receipts),
            "content_id": self.content_id,
            "content_reused": self.content_reused,
            "bloom_watermark": self.bloom_watermark,
            "similarity_candidates": list(self.similarity_candidates),
            "verification_required": self.verification_required,
            "errors": list(self.errors),
        }


class DataInfrastructurePipeline:
    """Assemble the Point 18 layered data infrastructure for one deployment."""

    def __init__(self, config: PipelineConfig | None = None) -> None:
        """Build every subsystem the configuration justifies."""
        self.config = config or PipelineConfig()
        cfg = self.config
        self.contracts = default_data_contracts()
        self.ingestion = self._build_ingestion()
        self.index_registry = DerivedIndexRegistry()
        self.metrics = DataInfrastructureMetrics()
        self.coordinator = MainBrainCoordinatorAdapter()
        self.cros = CROSDataBudgetAdapter()
        self.backpressure = BackpressureController(
            queue_limit=1024
        )
        self.workspace = WorkspaceDataPublisher()
        self.integrity = IntegrityVerifier()
        self.commitments: list[MerkleCommitment] = []

        # Approximate side structures (each registered as a derived index).
        self.bloom = BloomFilter(
            BloomParameters.from_cardinality_and_error(
                cfg.expected_membership_keys, cfg.bloom_target_fp_rate
            )
        )
        self.precheck_router = BloomPrecheckRouter(
            self.bloom, {}  # wired to the authoritative view lazily
        )
        self.count_min = CountMinSketch(
            CountMinParameters.from_error_bounds(
                cfg.count_min_epsilon, cfg.count_min_delta
            )
        )
        self.hyperloglog = HyperLogLog(cfg.hll_precision)
        self.kll = KLLQuantileSketch(k=cfg.kll_k)
        self.space_saving = SpaceSavingSummary(cfg.heavy_hitter_capacity)
        self.feature_schema = FeatureSchema(
            representation=FeatureRepresentation.TOKEN_SHINGLES,
            shingle_width=3,
            language="en",
            tokenizer_version="ws/1",
            normalization_version="nfkc-cf/1",
        )
        self.signer = MinHashSigner(
            self.feature_schema, permutations=cfg.minhash_permutations
        )
        self.lsh = LshBandIndex(
            choose_bands(cfg.minhash_permutations, cfg.similarity_threshold)
        )
        self.similarity_index = SimHashIndex()
        self._signatures: dict[str, MinHashSignature] = {}
        self._simhashes: dict[str, SimHashFingerprint] = {}

        # Durable storage adapters.
        root = Path(cfg.workspace_root or "ore-data-workspace")
        self.content_store = ContentAddressedStore(root / "objects")
        self.lsm: LsmStore | None = (
            LsmStore(root / "lsm") if cfg.lsm_justified else None
        )
        self.erasure_policy = (
            ErasurePolicy(8, 12, "up to 4 of 12 shards may be lost")
            if cfg.erasure_enabled
            else None
        )

        self.capabilities = DataInfrastructureCapabilities(
            bloom_active=True,
            count_min_active=True,
            hyperloglog_active=True,
            quantile_active=True,
            heavy_hitter_active=True,
            similarity_active=True,
            content_addressed_active=True,
            merkle_active=True,
            lsm_active=self.lsm is not None,
            erasure_active=self.erasure_policy is not None,
        )
        self._register_derived_indexes()
        self._events_since_rotation = 0

    # -- assembly ---------------------------------------------------------

    def _build_ingestion(self) -> IngestionService:
        """Build the Phase B ingestion service with its validation policy."""
        policy = ValidationPolicy(
            family_id="ore.workspace.findings",
            allowed_schema_versions=frozenset({"0.1.0"}),
            required_fields=("entity_key", "record_type"),
            max_record_bytes=1 << 20,
            authorized_sources=self.config.authorized_sources,
        )
        return IngestionService(policies={policy.family_id: policy})

    def _register_derived_indexes(self) -> None:
        """Register every derived structure with source, watermark and rebuild path (Step 12)."""
        from ore_data.contracts import FreshnessContract, UpdateMode

        specs = (
            DerivedIndexSpec(
                index_id="ore.data.index.bloom_membership",
                kind=IndexKind.BLOOM_FILTER,
                authoritative_source="ore_data.ingestion.AuthoritativeStore",
                schema_version="0.1.0",
                build_watermark=0,
                parameters=dict(self.bloom.parameters.to_dict()),
                checksum="registered-at-build",
                rebuild_procedure="rebuild from authoritative keys at watermark",
                memory_budget_bytes=self.bloom.memory_bytes(),
                freshness=FreshnessContract(
                    "bloom_membership", UpdateMode.QUEUED,
                    self.config.bloom_rebuild_interval_events,
                ),
            ),
            DerivedIndexSpec(
                index_id="ore.data.index.count_min_frequency",
                kind=IndexKind.COUNT_MIN_SKETCH,
                authoritative_source="ore_data.ingestion.AuthoritativeStore",
                schema_version="0.1.0",
                build_watermark=0,
                parameters=dict(self.count_min.parameters.to_dict()),
                checksum="registered-at-build",
                rebuild_procedure="reset counters from authoritative event replay",
                memory_budget_bytes=self.count_min.memory_bytes(),
                freshness=FreshnessContract(
                    "count_min_frequency", UpdateMode.QUEUED,
                    self.config.sketch_rotation_interval_events,
                ),
            ),
            DerivedIndexSpec(
                index_id="ore.data.index.hyperloglog_cardinality",
                kind=IndexKind.HYPERLOGLOG,
                authoritative_source="ore_data.ingestion.AuthoritativeStore",
                schema_version="0.1.0",
                build_watermark=0,
                parameters={"precision": self.hyperloglog.precision},
                checksum="registered-at-build",
                rebuild_procedure="re-add distinct items from authoritative replay",
                memory_budget_bytes=self.hyperloglog.memory_bytes(),
                freshness=FreshnessContract(
                    "hyperloglog_cardinality", UpdateMode.QUEUED,
                    self.config.sketch_rotation_interval_events,
                ),
            ),
            DerivedIndexSpec(
                index_id="ore.data.index.similarity_candidates",
                kind=IndexKind.MINHASH_LSH,
                authoritative_source="ore_data.ingestion.AuthoritativeStore",
                schema_version="0.1.0",
                build_watermark=0,
                parameters={
                    "permutations": self.config.minhash_permutations,
                    "bands": self.lsh.config.bands,
                    "rows_per_band": self.lsh.config.rows_per_band,
                },
                checksum="registered-at-build",
                rebuild_procedure="re-sign and re-bucket every authoritative record",
                memory_budget_bytes=1 << 20,
                freshness=FreshnessContract(
                    "similarity_candidates", UpdateMode.QUEUED, 5_000
                ),
            ),
        )
        for spec in specs:
            self.index_registry.register(spec)
            self.workspace.register_derived_view(spec.index_id, 0)

    # -- the end-to-end record path (source section 2) ------------------------

    def process_record(
        self,
        record: Mapping[str, Any],
        *,
        idempotency_key: str,
        provenance: ProvenanceInfo,
        text: str | None = None,
        observation: float | None = None,
        artifact: bytes | None = None,
        artifact_class: str = "accepted_algorithm",
    ) -> ProcessingOutcome:
        """Process one record through the end-to-end data pipeline (section 2).

        Validation, canonical identity, authoritative commit, event
        publication, storage routing, derived-index updates, streaming
        sketch updates, membership acceleration, and similarity candidate
        generation -- with verification always left explicit.
        """
        outcome = ProcessingOutcome(status="validated")
        family = str(record.get("family", "ore.workspace.findings"))
        if family not in self.ingestion.policies:
            outcome.status = "quarantined"
            outcome.errors = (f"no validation policy for family {family!r}",)
            self.ingestion.quarantine.hold(record, outcome.errors, family)
            return outcome
        receipt = self.ingestion.ingest(record, provenance, idempotency_key)
        if receipt is None:
            outcome.status = "rejected_or_quarantined"
            outcome.errors = (
                "record failed validation; it updated no index and no commitment",
            )
            return outcome

        outcome.commit_receipt = receipt
        outcome.entity_key = receipt.entity_key
        outcome.status = receipt.status.value
        entity_key = receipt.entity_key
        record_id = str(record.get("record_id", entity_key))
        outcome.record_id = record_id

        # Storage routing (Step 11): immutable artifacts go to content-addressed
        # storage; mutable state stays transactional; the LSM path stays off
        # until benchmarks justify it (Step 44).
        if artifact is not None:
            stored = self.content_store.put(
                artifact,
                logical_name=record_id,
                version=f"seq-{receipt.sequence}",
                artifact_class=artifact_class,
            )
            outcome.content_id = stored.content_id
            outcome.content_reused = stored.reused
            self.metrics.record_deduplication_ratio(
                self.content_store.deduplication_ratio()
            )
            self._publish_integrity_event(entity_key, stored.content_id)

        if self.lsm is not None:
            self.lsm.put(canonical_key("record", entity_key), canonicalize(record))

        # Duplicate deliveries update no derived structure: consumers detect
        # duplicate events and handle replay safely (Step 10).
        if receipt.status is not RecordStatus.COMMITTED:
            return outcome

        # Derived side indexes: bloom membership + watermark (Steps 16-17).
        # The pipeline updates the filter synchronously with the authoritative
        # commit, so it is complete for the watermark it was advanced to.
        self.bloom.add(entity_key, watermark=receipt.sequence)
        self.bloom.mark_complete(receipt.sequence)
        self.index_registry.update_watermark(
            "ore.data.index.bloom_membership", receipt.sequence
        )
        outcome.bloom_watermark = self.bloom.insertion_watermark

        # Streaming sketches: frequency, cardinality, quantiles, heavy hitters.
        self.count_min.update(record_id, watermark=receipt.sequence)
        self.index_registry.update_watermark(
            "ore.data.index.count_min_frequency", receipt.sequence
        )
        self.hyperloglog.add(record_id, watermark=receipt.sequence)
        self.index_registry.update_watermark(
            "ore.data.index.hyperloglog_cardinality", receipt.sequence
        )
        self.space_saving.update(record_id, watermark=receipt.sequence)
        if observation is not None:
            self.kll.update(observation, watermark=receipt.sequence)

        # Sketch rotation under the configured interval (Step 25).
        self._events_since_rotation += 1
        if self._events_since_rotation >= self.config.sketch_rotation_interval_events:
            self.count_min.rotate()
            self._events_since_rotation = 0

        # Similarity candidate generation, verification stays explicit (Step 40-42).
        if text is not None:
            signature = self.signer.signature_from_text(
                record_id, text, watermark=receipt.sequence
            )
            candidates = self.lsh.candidates(signature)
            self.lsh.add(signature)
            self._signatures[record_id] = signature
            fingerprint = simhash(
                record_id,
                {shingle: 1.0 for shingle in build_shingles(text, self.feature_schema)},
            )
            self.similarity_index.add(fingerprint)
            self._simhashes[record_id] = fingerprint
            outcome.similarity_candidates = sorted(candidates)
            outcome.verification_required = bool(candidates)
            self.index_registry.update_watermark(
                "ore.data.index.similarity_candidates", receipt.sequence
            )

        # Publish the change event to the workspace (Step 10 + section 6).
        for event in self.ingestion.store.outbox.pull(receipt.sequence - 1):
            if event.event_id == receipt.event_id:
                self.workspace.publish_change_event(event)
                outcome.event_receipts.append(
                    {
                        "event_id": event.event_id,
                        "sequence": event.sequence,
                        "event_type": event.event_type.value,
                    }
                )
        return outcome

    def _publish_integrity_event(self, entity_key: str, content: str) -> None:
        """Publish one integrity event for a stored artifact (Step 10)."""
        event = DataChangeEvent(
            event_id=f"integrity-{entity_key}-{content[:12]}",
            event_type=ChangeEventType.INTEGRITY,
            entity_key=entity_key,
            source="ore_data.content_store",
            schema_version="0.1.0",
            sequence=self.ingestion.store.watermark,
            payload_digest=content,
        )
        self.workspace.publish_change_event(event)

    # -- queries (two-stage membership, estimates, candidates) -------------------

    def membership(self, key: str) -> dict[str, object]:
        """Answer one membership query through the safe two-stage path (Step 17)."""
        watermark = self.ingestion.store.watermark
        hint = self.bloom.membership_hint(key, watermark)
        exact = self.ingestion.store.contains(key)
        self.coordinator.record_two_stage_lookup(
            shortcircuited=hint.verdict is MembershipVerdict.DEFINITELY_ABSENT
        )
        return {
            "key": key,
            "hint": hint.to_dict(),
            "exact_member": exact,
            "exactness": ExactnessClass.EXACT_MEMBERSHIP.value,
        }

    def frequency(self, key: str) -> FrequencyEstimate:
        """Return one frequency estimate for prioritization, not truth (Step 23-24)."""
        return self.count_min.estimate(key)

    def cardinality(self) -> CardinalityEstimate:
        """Return the distinct-record cardinality estimate (Step 30)."""
        return self.hyperloglog.estimate()

    def quantiles(self, percentiles: Sequence[float]) -> tuple[QuantileEstimate, ...]:
        """Return operational percentile estimates (Step 34)."""
        return tuple(self.kll.query(percentile) for percentile in percentiles)

    def heavy_hitters(self, top_k: int | None = None) -> tuple[HeavyHitterCandidate, ...]:
        """Return heavy-hitter candidates (Step 35)."""
        return self.space_saving.top(top_k)

    def similarity_candidates(self, record_id: str) -> list[str]:
        """Return LSH candidates for one record (Step 40); verification is separate."""
        signature = self._signatures.get(record_id)
        if signature is None:
            return []
        return sorted(self.lsh.candidates(signature))

    def verify_similarity(
        self,
        first_id: str,
        second_id: str,
        *,
        texts: Mapping[str, str],
        threshold: float | None = None,
    ) -> VerifiedSimilarity:
        """Verify one candidate pair by exact set similarity (Step 42)."""
        threshold = self.config.similarity_threshold if threshold is None else threshold
        first_signature = self._signatures.get(first_id)
        second_signature = self._signatures.get(second_id)
        first_features = set(
            build_shingles(texts[first_id], self.feature_schema)
            if first_id in texts
            else []
        )
        second_features = set(
            build_shingles(texts[second_id], self.feature_schema)
            if second_id in texts
            else []
        )
        candidate_score = None
        if first_signature is not None and second_signature is not None:
            candidate_score = estimated_jaccard(first_signature, second_signature).value
        similarity = verify_candidate_similarity(
            (first_id, second_id),
            candidate_score=candidate_score,
            first_features=first_features,
            second_features=second_features,
            threshold=threshold,
        )
        self.metrics.record_similarity_verification()
        classification = classify_duplicate(
            candidate_score=candidate_score,
            verified_similarity=similarity,
        )
        provenance = record_duplicate_provenance(
            pair=(first_id, second_id),
            classification=classification,
            feature_schema=self.feature_schema.schema_id(),
            thresholds={"similarity_threshold": threshold},
            candidate_score=candidate_score,
            exact_score=similarity.exact_value,
            verification_method=similarity.verification_method,
            verification_result=similarity.verified,
        )
        self.last_verification_provenance = provenance
        return similarity

    # -- integrity ----------------------------------------------------------------

    def build_integrity_snapshot(
        self,
        records: Sequence[bytes],
        snapshot_id: str,
    ) -> MerkleCommitment:
        """Build and publish one Merkle commitment over canonical records (Step 54)."""
        tree, commitment = build_merkle_snapshot(records, snapshot_id)
        self.integrity.publish_root(commitment)
        self.commitments.append(commitment)
        return commitment

    def verify_record(
        self,
        snapshot_id: str,
        record: bytes,
        leaf_index: int,
        tree: MerkleTree,
    ) -> dict[str, object]:
        """Verify one record against a snapshot's trusted root (Step 55)."""
        proof = tree.proof(leaf_index)
        result = self.integrity.verify(snapshot_id, record, proof)
        self.metrics.record_merkle(0.0, failed=not result.consistent)
        return result.to_dict()

    def archive(self, content: bytes) -> ErasureArchive | None:
        """Erasure-code one archive when the policy is enabled (Step 56)."""
        if self.erasure_policy is None:
            return None
        archive = ErasureArchive(self.erasure_policy)
        archive.encode(content)
        return archive

    # -- reconciliation and observability --------------------------------------------

    def reconcile(self) -> ReconciliationReport:
        """Reconcile every derived index against the authoritative watermark (Step 19)."""
        report = reconcile_derived_indexes(
            self.index_registry, self.ingestion.store.watermark
        )
        from ore_data.observability import metrics_from_reconciliation

        metrics_from_reconciliation(self.metrics, report)
        return report

    def index_health(self) -> dict[str, object]:
        """Return every derived index's freshness, watermark, version and health."""
        monitor = IndexHealthMonitor(self.index_registry)
        return monitor.report(self.ingestion.store.watermark)

    def telemetry(self, telemetry: StorageTelemetry) -> dict[str, int]:
        """Feed one storage telemetry sample to CROS bounded adaptation (section 6)."""
        return self.cros.adapt(telemetry)

    def metrics_snapshot(self) -> dict[str, object]:
        """Return the full metric snapshot (section 8.5)."""
        snapshot = self.metrics.snapshot()
        snapshot["coordinator"] = self.coordinator.stats()
        snapshot["backpressure"] = self.backpressure.stats()
        return snapshot

    def describe(self) -> dict[str, Any]:
        """Return the full assembly description of this deployment."""
        return {
            "contract": ORE_DATA_CONTRACT,
            "version": ORE_DATA_VERSION,
            "capabilities": self.capabilities.to_dict(),
            "bloom": self.bloom.describe(),
            "count_min": self.count_min.describe(),
            "hyperloglog": self.hyperloglog.describe(),
            "kll": self.kll.describe(),
            "heavy_hitters": self.space_saving.describe(),
            "content_store": self.content_store.stats(),
            "integrity_verifier": self.integrity.stats(),
            "derived_indexes": [spec.to_dict() for spec in self.index_registry.specs()],
            "operation_count": operation_count(),
            "phases": list(phases()),
        }

    @staticmethod
    def registry_summary() -> dict[str, Any]:
        """Return the 56-operation registry summary."""
        return {
            "operation_count": operation_count(),
            "phases": list(phases()),
            "operations": [
                {
                    "operation_id": spec.operation_id,
                    "phase": spec.phase,
                    "step": spec.source_step,
                    "name": spec.name,
                }
                for spec in OPERATION_REGISTRY
            ],
        }

    @staticmethod
    def interface_checks() -> dict[str, bool]:
        """Return the section 5 interface conformance results."""
        return verify_protocol_implementations()


def build_pipeline(
    *,
    workspace_root: Path | str | None = None,
    expected_membership_keys: int = 4096,
    bloom_target_fp_rate: float = 0.01,
    minhash_permutations: int = 64,
    similarity_threshold: float = 0.8,
    lsm_justified: bool = False,
    erasure_enabled: bool = False,
    authorized_sources: frozenset[str] | None = None,
) -> DataInfrastructurePipeline:
    """Build one data-infrastructure pipeline from explicit parameters."""
    config = PipelineConfig(
        workspace_root=Path(workspace_root) if workspace_root else None,
        expected_membership_keys=expected_membership_keys,
        bloom_target_fp_rate=bloom_target_fp_rate,
        minhash_permutations=minhash_permutations,
        similarity_threshold=similarity_threshold,
        lsm_justified=lsm_justified,
        erasure_enabled=erasure_enabled,
        authorized_sources=authorized_sources,
    )
    return DataInfrastructurePipeline(config)


__all__ = [
    "PipelineConfig",
    "DataInfrastructureCapabilities",
    "ProcessingOutcome",
    "DataInfrastructurePipeline",
    "build_pipeline",
    "estimate_metadata",
]
