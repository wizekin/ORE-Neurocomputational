"""Leakage acceptance tests 9-10: leakage-safe splits and filtered negative sampling.

9. Dataset splits follow the documented leakage policy.
10. Negative sampling filters known facts and documents its policy.
"""
from __future__ import annotations

import numpy as np

from ore_graph_intelligence.embeddings import (
    SPLIT_POLICIES,
    FilteredNegativeSampler,
    TripleRecord,
    build_split_manifest,
)


def _records():
    entities = [f"e{index:02d}" for index in range(10)]
    records = []
    for position, subject in enumerate(entities):
        obj = entities[(position + 4) % len(entities)]
        records.append(TripleRecord(
            subject=subject, relation="is_a", obj=obj,
            timestamp=f"2026-11-{position % 28 + 1:02d}",
            source_id=f"source:{position % 3}",
        ))
    return records


def test_dataset_split_follows_leakage_policy():
    """Test 9: entity-disjoint splits keep test entities out of training."""
    records = _records()
    manifest = build_split_manifest(records, policy="entity", seed=0)
    assert manifest.policy in SPLIT_POLICIES
    assert manifest.entity_disjoint(), "entity splits must not share entities"
    train_entities = {
        entity for subject, _relation, obj in manifest.train_triples
        for entity in (subject, obj)
    }
    test_entities = {
        entity for subject, _relation, obj in manifest.test_triples
        for entity in (subject, obj)
    }
    assert not (train_entities & test_entities), (
        "random edge splits leak through neighboring facts; entities must be disjoint"
    )
    assert manifest.notes, "the leakage policy must be documented in the manifest"
    assert manifest.train_digest and manifest.test_digest, "split hashes are recorded"


def test_negative_sampling_filters_known_facts():
    """Test 10: negatives exclude every triple known to train/validation/test."""
    records = _records()
    manifest = build_split_manifest(records, policy="relation", seed=0)
    known = set(manifest.train_triples) | set(manifest.validation_triples)
    sampler = FilteredNegativeSampler(known_triples=known)
    rng = np.random.default_rng(0)
    samples = sampler.sample(rng, count=32)
    assert samples, "the sampler must produce negatives"
    for sample in samples:
        assert sample not in known, (
            "an incomplete graph omits true facts; naive sampling creates false negatives"
        )
    documented = sampler.policy_description()
    assert documented and "filtered" in documented.lower(), (
        "the negative sampling policy must be documented"
    )


def test_time_aware_split_keeps_chronology():
    """Time-based splits put later triples into validation/test only."""
    records = _records()
    manifest = build_split_manifest(records, policy="time", seed=0)
    if manifest.train_triples and manifest.test_triples:
        train_times = {
            record.timestamp for record in records
            if record.triple() in set(manifest.train_triples)
        }
        test_times = {
            record.timestamp for record in records
            if record.triple() in set(manifest.test_triples)
        }
        if train_times and test_times:
            assert max(train_times) <= min(test_times), (
                "time leakage: training must not see the future"
            )
