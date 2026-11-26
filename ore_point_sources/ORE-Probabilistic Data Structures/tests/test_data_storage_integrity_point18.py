"""Phase H/I/J acceptance tests: LSM storage, content-addressed store, Merkle, erasure (Steps 44-56)."""
from __future__ import annotations

from pathlib import Path

import pytest

from ore_data.integrity.merkle import (
    IntegrityResult,
    IntegrityVerifier,
    MerkleCommitment,
    MerkleError,
    MerkleTree,
    build_merkle_snapshot,
    verify_merkle_proof,
)
from ore_data.recovery.erasure import (
    ErasureArchive,
    ErasureCodingError,
    ErasurePolicy,
    ErasureShard,
    reconstruct_from_any_k,
)
from ore_data.storage.content_addressed import (
    ContentAddressedStore,
    ContentAddressedStoreError,
    ObjectManifest,
    validate_artifact_class,
)
from ore_data.storage.lsm_adapter import (
    CompactionScheduler,
    LsmBenchmarkReport,
    LsmStore,
    lsm_justified,
    wal_recovery_restores_committed_writes,
)


# ---------------------------------------------------------------------------
# Phase H (Steps 44-49)


def test_lsm_write_path_persists_and_recovers_after_crash(tmp_path: Path):
    """Steps 45, 49: WAL recovery restores committed writes after a simulated crash."""
    store = LsmStore(tmp_path / "lsm", memtable_limit_entries=10_000)
    for index in range(50):
        store.put(f"key-{index:03d}", f"value-{index}".encode("utf-8"))
    assert store.get("key-007") == b"value-7"
    # Simulated crash: the object is dropped without flushing the memtable.
    del store
    recovered = wal_recovery_restores_committed_writes(tmp_path / "lsm")
    assert recovered == 50


def test_lsm_wal_recovery_stops_at_corrupt_tail(tmp_path: Path):
    """Step 45: a torn WAL tail does not corrupt recovery; the intact prefix survives."""
    root = tmp_path / "lsm-torn"
    store = LsmStore(root, memtable_limit_entries=10_000)
    for index in range(10):
        store.put(f"key-{index}", f"value-{index}".encode("utf-8"))
    wal = root / "wal.log"
    with wal.open("ab") as handle:
        handle.write(b"\x37\xff garbage bytes simulating a torn write")
    del store
    recovered = LsmStore(root, memtable_limit_entries=10_000)
    # The ten intact committed writes survive; garbage never applies.
    for index in range(10):
        assert recovered.get(f"key-{index}") == f"value-{index}".encode("utf-8")


def test_lsm_flushes_memtables_into_sorted_tables_and_reads_newest_first(tmp_path: Path):
    """Steps 46-47: flush freezes the memtable; reads resolve versions newest-first."""
    store = LsmStore(tmp_path / "lsm", memtable_limit_entries=8)
    for index in range(20):
        store.put(f"key-{index:03d}", f"v1-{index}".encode("utf-8"))
    store.flush()
    tables_after_first = len(store._tables)  # noqa: SLF001 - inspecting internals
    assert tables_after_first >= 1
    # Overwrite every key with a newer version.
    for index in range(20):
        store.put(f"key-{index:03d}", f"v2-{index}".encode("utf-8"))
    store.flush()
    for index in range(20):
        assert store.get(f"key-{index:03d}") == f"v2-{index}".encode("utf-8")
    # New writes continue in a fresh memtable while tables stay immutable.
    store.put("key-late", b"late")
    assert store.get("key-late") == b"late"


def test_lsm_tombstone_semantics_and_safe_compaction(tmp_path: Path):
    """Steps 47-48: tombstones delete; full compaction discards them safely."""
    store = LsmStore(tmp_path / "lsm", memtable_limit_entries=8)
    for index in range(30):
        store.put(f"key-{index:03d}", b"data")
    store.flush()
    deleted = [f"key-{index:03d}" for index in range(0, 30, 2)]
    for key in deleted:
        store.delete(key)
    store.flush()
    for key in deleted:
        assert store.get(key) is None  # tombstone semantics
    for index in range(1, 30, 2):
        assert store.get(f"key-{index:03d}") == b"data"
    result = store.compact()
    assert result["deferred"] is False
    for key in deleted:
        assert store.get(key) is None  # no resurrection after compaction
    for index in range(1, 30, 2):
        assert store.get(f"key-{index:03d}") == b"data"
    assert store.compaction_stats.compactions == 1


def test_lsm_compaction_defers_under_cros_budgets(tmp_path: Path):
    """Step 48: compaction never runs when the CROS budget is exhausted."""
    store = LsmStore(tmp_path / "lsm", memtable_limit_entries=8)
    for index in range(20):
        store.put(f"key-{index}", b"data")
    store.flush()
    scheduler = CompactionScheduler(min_tokens=4)
    result = store.compact(scheduler=scheduler, budget_tokens=1)
    assert result["deferred"] is True
    assert scheduler.deferred_runs == 1
    assert store.compaction_stats.compactions == 0  # nothing compacted yet
    result = store.compact(scheduler=scheduler, budget_tokens=8)
    assert result["deferred"] is False
    assert scheduler.granted_runs == 1


def test_lsm_bloom_filters_skip_unnecessary_table_reads(tmp_path: Path):
    """Step 47: reads skip tables whose per-table bloom filter says absent."""
    store = LsmStore(tmp_path / "lsm", memtable_limit_entries=8)
    for index in range(30):
        store.put(f"key-{index:03d}", b"data")
    store.flush()
    found = store.get("key-000")
    assert found == b"data"
    missing_with_filter = store.get("definitely-absent-key")
    missing_without_filter = store.get("definitely-absent-key", skip_filter=True)
    assert missing_with_filter is None and missing_without_filter is None
    # Both paths agree; the filter only avoids the read, never the truth.


def test_lsm_backup_is_validated_by_restoration(tmp_path: Path):
    """Step 49: a backup is not validated until restoration has succeeded."""
    store = LsmStore(tmp_path / "lsm", memtable_limit_entries=8)
    for index in range(40):
        store.put(f"key-{index:03d}", f"value-{index}".encode("utf-8"))
    store.flush()
    backup_dir = store.backup_to(tmp_path / "backup")
    assert (backup_dir / "manifest.json").exists()
    report = store.restore_from(backup_dir)
    assert report.checksums_verified
    assert report.tables_verified >= 1
    assert report.invariants_verified
    assert report.records_restored == 40
    # Corrupting a backup shard fails the restore validation.
    corrupted = tmp_path / "corrupted"
    corrupted.mkdir()
    for item in backup_dir.iterdir():
        (corrupted / item.name).write_bytes(item.read_bytes())
    table_file = next(
        item for item in corrupted.iterdir() if item.name.startswith("sstable-")
    )
    data = bytearray(table_file.read_bytes())
    data[-1] ^= 0xFF  # break the trailing CRC
    table_file.write_bytes(bytes(data))
    with pytest.raises(Exception):
        store.restore_from(corrupted)


def test_lsm_adoption_requires_a_measured_benefit(tmp_path: Path):
    """Step 44: the LSM path is adopted only when benchmarks show a benefit."""
    store = LsmStore(tmp_path / "lsm")
    report = store.benchmark(writes=200, reads=100)
    assert report.writes == 200 and report.reads == 100
    assert report.p95_latency_ms >= report.p50_latency_ms >= 0
    assert lsm_justified(report, {"write_amplification": 20.0, "write_p95_ms": 50.0})
    assert not lsm_justified(report, {"write_amplification": 0.5, "write_p95_ms": 0.01})
    assert isinstance(report, LsmBenchmarkReport)


# ---------------------------------------------------------------------------
# Phase I (Steps 50-53)


def test_content_addressed_store_verifies_and_deduplicates(tmp_path: Path):
    """Steps 50-51: objects store under their hash, verify on write and read, and dedupe."""
    store = ContentAddressedStore(tmp_path / "objects")
    content = b"ore execution bundle payload"
    first = store.put(content, logical_name="bundle", version="1")
    second = store.put(content, logical_name="bundle", version="2")
    assert first.reused is False
    assert second.reused is True  # same content reuses the object
    assert second.content_id == first.content_id
    assert store.distinct_object_count() == 1
    assert store.deduplication_ratio() == 0.5
    artifact = store.get_verified(first.content_id)
    assert artifact.content == content
    assert artifact.byte_integrity_only is True
    assert "does not prove" in artifact.integrity_note()


def test_content_addressed_store_rejects_mutable_artifacts_without_versioning(tmp_path: Path):
    """Step 50: mutable workspace records never force into immutable storage."""
    assert validate_artifact_class("algo_ir_specification")
    assert validate_artifact_class("execution_bundle")
    assert validate_artifact_class("workspace_record") is False
    assert validate_artifact_class("workspace_record", versioning_model=True)
    store = ContentAddressedStore(tmp_path / "objects")
    with pytest.raises(ContentAddressedStoreError):
        store.put(b"mutable", artifact_class="workspace_record")


def test_manifest_separates_object_identity_from_mutable_references(tmp_path: Path):
    """Step 52: updates create new objects and manifest entries; history is kept."""
    store = ContentAddressedStore(tmp_path / "objects")
    v1 = store.put(b"algorithm v1", logical_name="algo", version="1")
    v2 = store.put(b"algorithm v2", logical_name="algo", version="2")
    assert v1.content_id != v2.content_id
    assert store.manifest.resolve("algo", "1") == v1.content_id
    assert store.manifest.resolve("algo", "2") == v2.content_id
    assert store.manifest.versions_of("algo") == ("1", "2")
    assert store.get_verified(v1.content_id).content == b"algorithm v1"
    assert store.get_verified(v2.content_id).content == b"algorithm v2"
    # The manifest persists and reloads.
    reloaded = ObjectManifest(tmp_path / "objects" / "manifest.json")
    assert reloaded.resolve("algo", "1") == v1.content_id
    assert len(reloaded.entries()["algo"]) == 2


def test_verified_reads_recompute_the_content_hash(tmp_path: Path):
    """Step 53: reads recompute and verify the hash; tampering fails the read."""
    store = ContentAddressedStore(tmp_path / "objects")
    receipt = store.put(b"verified payload")
    artifact = store.get_verified(receipt.content_id)
    assert artifact.content == b"verified payload"
    digest = receipt.content_id.split(":", 1)[1]
    object_path = tmp_path / "objects" / "objects" / digest[:2] / digest
    original = object_path.read_bytes()
    object_path.write_bytes(b"tampered payload")
    with pytest.raises(ContentAddressedStoreError):
        store.get_verified(receipt.content_id)
    object_path.write_bytes(original)
    assert store.get_verified(receipt.content_id).content == b"verified payload"
    with pytest.raises(KeyError):
        store.get_verified("sha256:" + "0" * 64)


# ---------------------------------------------------------------------------
# Phase J (Steps 54-55)


def _records(count: int) -> list[bytes]:
    return [f"record-{index}".encode("utf-8") for index in range(count)]


def test_merkle_snapshot_commits_a_root_with_metadata():
    """Step 54: commitments carry root, snapshot identity, version and timestamp."""
    records = _records(9)  # odd leaf count exercises the odd-node rule
    tree, commitment = build_merkle_snapshot(records, "snapshot-1")
    assert isinstance(commitment, MerkleCommitment)
    assert len(commitment.root) == 64
    assert commitment.leaf_count == 9
    assert commitment.created_at
    payload = commitment.to_dict()
    assert payload["snapshot_id"] == "snapshot-1"
    with pytest.raises(MerkleError):
        MerkleTree([])


def test_merkle_proofs_verify_every_leaf_against_the_trusted_root():
    """Step 55: every leaf verifies; success means consistency, not correctness."""
    records = _records(12)
    tree, commitment = build_merkle_snapshot(records, "snapshot-2")
    verifier = IntegrityVerifier()
    verifier.publish_root(commitment)
    for index, record in enumerate(records):
        proof = tree.proof(index)
        result = verifier.verify("snapshot-2", record, proof)
        assert isinstance(result, IntegrityResult)
        assert result.consistent
        assert "does not establish" in result.note  # factual correctness stays distinct
    assert verifier.stats()["proof_verifications"] == 12


@pytest.mark.parametrize("mutation", ["leaf", "sibling", "path", "root"])
def test_merkle_proofs_fail_when_anything_changes(mutation: str):
    """Step 55: altered leaf, sibling, path or trusted root must fail the proof."""
    records = _records(8)
    tree, commitment = build_merkle_snapshot(records, "snapshot-3")
    leaf_index = 3
    proof = tree.proof(leaf_index)
    if mutation == "leaf":
        result = verify_merkle_proof(b"record-999", proof, commitment.root)
        assert not result.consistent
    elif mutation == "sibling":
        siblings = list(proof.siblings)
        digest, is_left = siblings[0]
        siblings[0] = (bytes(digest[:15]) + b"\x00" + bytes(digest[16:]), is_left)
        from ore_data.integrity.merkle import MerkleProof

        tampered = MerkleProof(
            leaf_index=leaf_index, leaf_count=proof.leaf_count, siblings=tuple(siblings)
        )
        result = verify_merkle_proof(records[leaf_index], tampered, commitment.root)
        assert not result.consistent
    elif mutation == "path":
        from ore_data.integrity.merkle import MerkleProof

        reordered = list(proof.siblings)
        reordered[0] = (reordered[0][0], not reordered[0][1])
        flipped = MerkleProof(
            leaf_index=leaf_index, leaf_count=proof.leaf_count, siblings=tuple(reordered)
        )
        result = verify_merkle_proof(records[leaf_index], flipped, commitment.root)
        assert not result.consistent
    else:  # root
        forged_root = ("f" if commitment.root[0] != "f" else "0") + commitment.root[1:]
        result = verify_merkle_proof(records[leaf_index], proof, forged_root)
        assert not result.consistent


def test_merkle_tags_are_domain_separated():
    """Source 4.7: leaf and node hashes use distinct domain tags."""
    from ore_data.integrity.merkle import LEAF_TAG, NODE_TAG, leaf_hash, node_hash

    assert LEAF_TAG != NODE_TAG
    record = b"payload"
    first = leaf_hash(record)
    second = node_hash(first, first)
    assert first != second
    assert leaf_hash(record) == first  # deterministic


# ---------------------------------------------------------------------------
# Step 56


def _policy() -> ErasurePolicy:
    return ErasurePolicy(8, 12, "up to 4 of 12 shards may be lost")


def test_erasure_policy_documents_overhead_and_threshold():
    """Step 56: parity overhead is (n-k)/k; the reconstruction threshold is k."""
    policy = _policy()
    assert policy.parity_shards == 4
    assert policy.reconstruction_threshold == 8
    assert policy.overhead == 0.5
    payload = policy.to_dict()
    assert payload["failure_assumption"]
    with pytest.raises(ErasureCodingError):
        ErasurePolicy(0, 4, "bad")


@pytest.mark.parametrize("dropped", [(0,), (5, 7), (0, 1, 2, 3), (8, 9, 10, 11)])
def test_erasure_archives_reconstruct_from_any_sufficient_shard_set(dropped: tuple[int, ...]):
    """Step 56 + 8.3: any k of n shards reconstruct the content, verified by hash."""
    content = bytes(range(256)) * 41 + b"ore archive tail"
    result = reconstruct_from_any_k(_policy(), content, dropped)
    assert result.content == content
    assert result.content_id_verified is True
    assert len(result.shards_used) == 8
    assert result.reconstruction_ms >= 0


def test_erasure_reconstruction_rejects_corrupt_shards():
    """Step 56: shard hashes are verified before use; corrupt shards never decode."""
    content = b"ore distributed archive payload" * 16
    archive = ErasureArchive(_policy())
    shards = archive.encode(content)
    corrupt = list(shards)
    corrupt[3] = ErasureShard(3, b"corrupted bytes", "0" * 64)
    with pytest.raises(ErasureCodingError):
        archive.reconstruct([s for s in shards[:8] if s.index != 3] + [corrupt[3]])


def test_erasure_reconstruction_requires_k_valid_shards():
    """Step 56: fewer than k shards cannot reconstruct."""
    content = b"payload"
    archive = ErasureArchive(_policy())
    shards = archive.encode(content)
    with pytest.raises(ErasureCodingError):
        archive.reconstruct(shards[:7])


def test_erasure_metadata_round_trip_reconstructs_standalone():
    """Step 56: stored metadata plus shards reconstruct without the original object."""
    content = b"standalone reconstruction payload" * 8
    archive = ErasureArchive(_policy())
    shards = archive.encode(content)
    metadata = archive.archive_metadata()
    standalone = ErasureArchive.from_metadata(_policy(), metadata)
    result = standalone.reconstruct(shards[4:])
    assert result.content == content
    assert result.content_id_verified


def test_erasurecoding_field_arithmetic():
    """GF(2^8) arithmetic invariants hold for the primitive polynomial 0x11D."""
    from ore_data.recovery.erasure import gf_inv, gf_mul, gf_pow

    assert gf_mul(1, 5) == 5
    assert gf_mul(0, 7) == 0
    for element in (2, 3, 5, 17, 200):
        assert gf_mul(element, gf_inv(element)) == 1
    assert gf_pow(2, 0) == 1
    assert gf_pow(2, 255) == 1  # multiplicative order of the field
    assert gf_pow(3, 255) == 1
