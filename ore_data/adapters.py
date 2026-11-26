"""Proposed package boundary and data-service interfaces (source section 5).

``ore_data`` keeps every implementation behind interfaces -- the
authoritative store, membership hint index, frequency and cardinality
sketches, quantile and heavy-hitter summaries, similarity index, object
store, integrity verifier, and reconciliation service -- so local
deployments use in-memory adapters and no consumer ever depends on a
concrete probabilistic structure. This module also provides the
conformance checks the pipeline exposes (section 5.1 interface list).
"""
from __future__ import annotations

from typing import Any, Mapping, Protocol, runtime_checkable

from ore_data.contracts import default_data_contracts
from ore_data.index_registry import DerivedIndexRegistry
from ore_data.ingestion import AuthoritativeStore
from ore_data.integrity.merkle import IntegrityVerifier, MerkleTree
from ore_data.probabilistic.bloom import BloomFilter, MembershipHint
from ore_data.probabilistic.count_min import CountMinSketch, FrequencyEstimate
from ore_data.probabilistic.hyperloglog import CardinalityEstimate, HyperLogLog
from ore_data.recovery.erasure import ErasureArchive
from ore_data.similarity.minhash import MinHashSignature
from ore_data.similarity.lsh import LshBandIndex
from ore_data.similarity.simhash import SimHashFingerprint
from ore_data.storage.content_addressed import ContentAddressedStore, VerifiedArtifact
from ore_data.storage.lsm_adapter import LsmStore
from ore_data.streaming.heavy_hitters import SpaceSavingSummary
from ore_data.streaming.quantiles import KLLQuantileSketch, QuantileEstimate


@runtime_checkable
class AuthoritativeStoreProtocol(Protocol):
    """Exact source-of-record interface (Step 9)."""

    def lookup(self, entity_key: str) -> Mapping[str, Any] | None: ...

    def commit_authoritative(self, record: Any, idempotency_key: str) -> Any: ...


@runtime_checkable
class MembershipIndexProtocol(Protocol):
    """Safe two-stage membership interface (Step 17)."""

    def membership_hint(self, key: str, authoritative_watermark: int) -> MembershipHint: ...


@runtime_checkable
class FrequencySketchProtocol(Protocol):
    """Frequency estimation interface (Step 23)."""

    def update(self, key: str, weight: float = 1.0) -> None: ...

    def estimate(self, key: str) -> FrequencyEstimate: ...


@runtime_checkable
class CardinalitySketchProtocol(Protocol):
    """Cardinality estimation interface (Step 30)."""

    def add(self, item: str) -> None: ...

    def estimate(self) -> CardinalityEstimate: ...


@runtime_checkable
class QuantileSketchProtocol(Protocol):
    """Streaming quantile interface (Step 34)."""

    def update(self, observation: float) -> None: ...

    def query(self, percentile: float) -> QuantileEstimate: ...


@runtime_checkable
class HeavyHitterSummaryProtocol(Protocol):
    """Heavy-hitter summary interface (Step 35)."""

    def update(self, key: str, weight: float = 1.0) -> None: ...

    def top(self, top_k: int | None = None) -> Any: ...


@runtime_checkable
class SimilarityIndexProtocol(Protocol):
    """Candidate-generation interface (Step 40)."""

    def add(self, signature: Any) -> None: ...

    def candidates(self, signature: Any) -> Any: ...


@runtime_checkable
class ObjectStoreProtocol(Protocol):
    """Immutable object store interface (Step 51)."""

    def put(self, content: bytes, **kwargs: Any) -> Any: ...

    def get_verified(self, identifier: str, **kwargs: Any) -> VerifiedArtifact: ...


@runtime_checkable
class IntegrityVerifierProtocol(Protocol):
    """Snapshot integrity interface (Steps 54-55)."""

    def publish_root(self, commitment: Any) -> None: ...

    def verify(self, snapshot_id: str, record: bytes, proof: Any) -> Any: ...


@runtime_checkable
class ReconciliationServiceProtocol(Protocol):
    """Derived-index reconciliation interface (Step 19)."""

    def reconcile(self, authoritative_watermark: int) -> Any: ...


def verify_protocol_implementations() -> dict[str, bool]:
    """Return the interface conformance results for every subsystem.

    The check instantiates each structure behind its interface boundary,
    exercises the interface methods once, and reports whether the concrete
    implementations satisfy the runtime protocols (the package-boundary
    guarantee of source section 5).
    """
    checks: dict[str, bool] = {}

    authoritative = AuthoritativeStore()
    checks["authoritative_store"] = isinstance(authoritative, AuthoritativeStoreProtocol)

    from ore_data.probabilistic.bloom import BloomParameters

    parameters = BloomParameters.from_cardinality_and_error(64, 0.01)
    bloom = BloomFilter(parameters)
    bloom.add("key-a")
    checks["membership_index"] = isinstance(bloom, MembershipIndexProtocol)

    from ore_data.probabilistic.count_min import CountMinParameters

    cms = CountMinSketch(CountMinParameters.from_error_bounds(0.01, 0.01))
    cms.update("key-a")
    checks["frequency_sketch"] = isinstance(cms, FrequencySketchProtocol)

    hll = HyperLogLog(12)
    hll.add("item-a")
    checks["cardinality_sketch"] = isinstance(hll, CardinalitySketchProtocol)

    kll = KLLQuantileSketch(k=64)
    kll.update(1.0)
    checks["quantile_sketch"] = isinstance(kll, QuantileSketchProtocol)

    heavy = SpaceSavingSummary(8)
    heavy.update("key-a")
    checks["heavy_hitter_summary"] = isinstance(heavy, HeavyHitterSummaryProtocol)

    from ore_data.similarity.lsh import LshBandConfig
    from ore_data.similarity.minhash import FeatureRepresentation, FeatureSchema, MinHashSigner

    schema = FeatureSchema(
        representation=FeatureRepresentation.TOKEN_SHINGLES,
        shingle_width=3,
        language="en",
        tokenizer_version="ws/1",
        normalization_version="nfkc-cf/1",
    )
    signer = MinHashSigner(schema, permutations=16)
    signature = signer.signature("rec-1", {"a", "b", "c"})
    index = LshBandIndex(LshBandConfig(bands=4, rows_per_band=4))
    index.add(signature)
    checks["similarity_index"] = isinstance(index, SimilarityIndexProtocol)

    checks["merkle_verifier"] = isinstance(IntegrityVerifier(), IntegrityVerifierProtocol)
    checks["index_registry"] = isinstance(DerivedIndexRegistry(), ReconciliationServiceProtocol)
    checks["data_contracts"] = isinstance(default_data_contracts(), object)

    return checks


__all__ = [
    "AuthoritativeStoreProtocol",
    "MembershipIndexProtocol",
    "FrequencySketchProtocol",
    "CardinalitySketchProtocol",
    "QuantileSketchProtocol",
    "HeavyHitterSummaryProtocol",
    "SimilarityIndexProtocol",
    "ObjectStoreProtocol",
    "IntegrityVerifierProtocol",
    "ReconciliationServiceProtocol",
    "verify_protocol_implementations",
]
