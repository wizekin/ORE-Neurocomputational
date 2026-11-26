"""End-to-end demonstration of the Point 18 layered data infrastructure.

The example walks one deployment through the full stack: canonical
validation and authoritative commit with idempotent retries, watermark-gated
Bloom membership, Count-Min/HLL/KLL/Space-Saving streaming analytics,
MinHash/LSH candidate nomination with exact verification, content-addressed
artifact storage with deduplication, Merkle commitments with proofs,
reconciliation of derived indexes, CROS telemetry adaptation and bounded
backpressure.
"""
from __future__ import annotations

from ore_data.controller_adapter import (
    BackpressureController,
    CROSDataBudgetAdapter,
    StorageTelemetry,
)
from ore_data.ingestion import ProvenanceInfo
from ore_data.pipeline import build_pipeline


def _record(key: str, record_type: str = "CONCEPT") -> dict[str, object]:
    return {
        "schema_version": "0.1.0",
        "entity_key": key,
        "record_type": record_type,
        "family": "ore.workspace.findings",
    }


def main() -> None:
    """Run the demonstration end to end."""
    print("== ORE Point 18: layered data infrastructure demonstration ==")

    # Source section 2: the end-to-end record path.
    pipeline = build_pipeline(workspace_root="ore-data-demo")
    provenance = ProvenanceInfo(source="ore_demo", actor="agent-1")

    text_a = "the quick brown fox jumps over the lazy dog again and again"
    text_b = "the quick brown fox jumps over the lazy dog again and more"
    unrelated = "completely unrelated words about distributed storage systems"

    first = pipeline.process_record(
        _record("concept-a"), idempotency_key="id-a",
        provenance=provenance, text=text_a, observation=11.0,
    )
    retry = pipeline.process_record(
        _record("concept-a"), idempotency_key="id-a",  # retry, applies once
        provenance=provenance, text=text_a, observation=11.0,
    )
    second = pipeline.process_record(
        _record("concept-b"), idempotency_key="id-b",
        provenance=provenance, text=text_b, observation=13.0,
    )
    third = pipeline.process_record(
        _record("concept-c"), idempotency_key="id-c",
        provenance=provenance, text=unrelated, observation=17.0,
    )
    print("commit status:", first.status, "| retry:", retry.status,
          "| sequence:", first.commit_receipt.sequence if first.commit_receipt else None)

    # Phase C: watermark-gated two-stage membership.
    present = pipeline.membership("concept-a")
    absent = pipeline.membership("missing-key")
    print("membership:", present["hint"]["verdict"],
          "| absent:", absent["hint"]["verdict"],
          "| skip safe:", absent["hint"]["safe_to_skip_exact_lookup"])

    # Phases D-F: streaming analytics as typed estimates, never truth.
    frequency = pipeline.frequency("concept-a")
    cardinality = pipeline.cardinality()
    quantiles = pipeline.quantiles((0.5, 0.9, 0.99))
    heavy = pipeline.heavy_hitters(3)
    print("frequency estimate:", frequency.estimate, "exact:", frequency.exact)
    print("cardinality estimate:", round(cardinality.estimate, 1),
          "±", round(cardinality.relative_standard_error, 4))
    print("p50/p90/p99:", [round(q.value, 1) for q in quantiles])
    print("heavy hitters:", [(c.key, c.estimate) for c in heavy])

    # Phase G: LSH candidates are nominated, then verified exactly.
    print("lsh candidates for concept-b:", second.similarity_candidates)
    verification = pipeline.verify_similarity(
        "concept-a", "concept-b",
        texts={"concept-a": text_a, "concept-b": text_b},
    )
    print("verified jaccard:", round(verification.exact_value or 0.0, 3),
          "| verified:", verification.verified)

    # Phase I: immutable artifacts route to content-addressed storage.
    payload = b"validated algoir specification payload"
    artifact = pipeline.process_record(
        _record("artifact-1", record_type="ALGOIR"), idempotency_key="art-1",
        provenance=provenance, artifact=payload,
        artifact_class="algo_ir_specification",
    )
    duplicate = pipeline.process_record(
        _record("artifact-2", record_type="ALGOIR"), idempotency_key="art-2",
        provenance=provenance, artifact=payload,
        artifact_class="algo_ir_specification",
    )
    print("artifact content id:", artifact.content_id[:16], "...",
          "| deduplicated:", duplicate.content_reused)

    # Phase J: Merkle commitments and proof verification.
    records = [f"record-{index}".encode("utf-8") for index in range(7)]
    from ore_data.integrity.merkle import MerkleTree

    commitment = pipeline.build_integrity_snapshot(records, "snap-demo")
    tree = MerkleTree(records)
    verified = pipeline.verify_record("snap-demo", records[4], 4, tree)
    tampered = pipeline.verify_record("snap-demo", b"record-999", 4, tree)
    print("merkle root:", commitment.root[:16], "...",
          "| proof:", verified["consistent"], "| tampered:", tampered["consistent"])

    # Steps 19 and 12: reconciliation and index health.
    health = pipeline.index_health()
    report = pipeline.reconcile()
    print("index health:", {k: v["health"] for k, v in health.items()})
    print("reconciliation: authoritative watermark", report.authoritative_watermark,
          "| stale:", report.indexes_stale)

    # Section 6: CROS bounded adaptation and backpressure.
    cros = CROSDataBudgetAdapter()
    adapted = cros.adapt(StorageTelemetry(
        memory_pressure=0.9, storage_pressure=0.9,
        queue_depth=50, compaction_load=0.8,
    ))
    print("cros adapted parameters:", adapted)

    controller = BackpressureController(queue_limit=4)
    for event in range(5):
        controller.offer(f"event-{event}")
    print("backpressure: queued", controller.depth(),
          "| deferred", controller.deferred)

    print("== demonstration complete ==")


if __name__ == "__main__":
    main()
