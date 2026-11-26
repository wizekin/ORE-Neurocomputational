"""Phase G acceptance tests: MinHash, LSH, SimHash, verification and provenance (Steps 37-43)."""
from __future__ import annotations

import random

import pytest

from ore_data.similarity.lsh import (
    LshBandConfig,
    LshBandIndex,
    candidate_probability,
    choose_bands,
    measure_recall,
    threshold_config_for,
)
from ore_data.similarity.minhash import (
    DEFAULT_PERMUTATIONS,
    DuplicateClassification,
    FeatureRepresentation,
    FeatureSchema,
    MinHashSigner,
    SimilarityEstimate,
    VerifiedSimilarity,
    build_shingles,
    classify_duplicate,
    estimated_jaccard,
    exact_jaccard,
    normalize_tokens,
    record_duplicate_provenance,
    verify_byte_identity,
    verify_candidate_similarity,
    verify_structured_fields,
)
from ore_data.similarity.simhash import (
    SimHashFingerprint,
    SimHashIndex,
    hamming_distance,
    simhash,
    validate_distance_threshold,
)


def _schema() -> FeatureSchema:
    return FeatureSchema(
        representation=FeatureRepresentation.TOKEN_SHINGLES,
        shingle_width=3,
        language="en",
        tokenizer_version="ws/1",
        normalization_version="nfkc-cf/1",
    )


# ---------------------------------------------------------------------------
# Steps 37-39


def test_normalization_and_shingles_are_canonical():
    """Steps 37-38: normalization is stable; shingle width follows the schema."""
    tokens = normalize_tokens("  The  QUICK brown Fox ")
    assert tokens == ("the", "quick", "brown", "fox")
    schema = _schema()
    shingles = build_shingles("the quick brown fox jumps", schema)
    assert shingles == {
        "the quick brown",
        "quick brown fox",
        "brown fox jumps",
    }
    short = build_shingles("one two", schema)
    assert short == {"one two"}
    payload = schema.to_dict()
    assert payload["shingle_width"] == 3
    assert payload["language"] == "en"


def test_minhash_estimates_jaccard_within_tolerance():
    """Step 39: the matching fraction estimates Jaccard with variance ~ J(1-J)/r."""
    rng = random.Random(23)
    universe = [f"feature-{i}" for i in range(400)]
    signer = MinHashSigner(_schema(), permutations=DEFAULT_PERMUTATIONS)
    errors = []
    for trial in range(24):
        first = set(rng.sample(universe, 200))
        shared = rng.sample(sorted(first), rng.randrange(0, 200))
        second = set(shared) | set(
            rng.sample([f for f in universe if f not in first], 200 - len(shared))
        )
        signature_a = signer.signature("a", first)
        signature_b = signer.signature("b", second)
        estimate = estimated_jaccard(signature_a, signature_b)
        assert isinstance(estimate, SimilarityEstimate)
        assert estimate.exact is False
        exact = exact_jaccard(first, second)
        errors.append(abs(estimate.value - exact))
    assert sum(errors) / len(errors) < 0.08


def test_minhash_signatures_require_comparable_schemas():
    """Step 38: signatures from different schemas never compare."""
    other = FeatureSchema(
        representation=FeatureRepresentation.TOKEN_SHINGLES,
        shingle_width=2,
        language="en",
        tokenizer_version="ws/1",
        normalization_version="nfkc-cf/1",
    )
    signer_a = MinHashSigner(_schema(), permutations=32)
    signer_b = MinHashSigner(other, permutations=32)
    signature_a = signer_a.signature("a", {"x", "y"})
    signature_b = signer_b.signature("b", {"x", "y"})
    with pytest.raises(ValueError):
        estimated_jaccard(signature_a, signature_b)
    same = signer_a.signature("c", {"x", "y"})
    assert estimated_jaccard(signature_a, same).value == 1.0
    with pytest.raises(ValueError):
        signer_a.signature("empty", set())


# ---------------------------------------------------------------------------
# Step 40


def test_lsh_candidate_probability_follows_the_banding_formula():
    """Step 40: P(candidate) = 1 - (1 - s^r)^b."""
    config = LshBandConfig(bands=16, rows_per_band=4)
    assert config.signature_length == 64
    assert candidate_probability(0.0, config) == 0.0
    assert candidate_probability(1.0, config) == 1.0
    s = 0.8
    expected = 1.0 - (1.0 - s**4) ** 16
    assert abs(candidate_probability(s, config) - expected) < 1e-12
    with pytest.raises(ValueError):
        candidate_probability(1.5, config)


def test_lsh_bucketing_generates_candidates_for_similar_records():
    """Step 40: similar records share bands; dissimilar records rarely do."""
    signer = MinHashSigner(_schema(), permutations=64)
    config = LshBandConfig(bands=16, rows_per_band=4)
    index = LshBandIndex(config)
    base = [f"token-{i}" for i in range(60)]
    similar = set(base[:55]) | {f"extra-{i}" for i in range(5)}
    other = {f"other-{i}" for i in range(60)}
    signature_base = signer.signature("base", set(base))
    signature_similar = signer.signature("similar", similar)
    signature_other = signer.signature("other", other)
    index.add_all([signature_base, signature_similar, signature_other])
    candidates = index.candidates(signature_base)
    assert "similar" in candidates  # high-Jaccard pair becomes a candidate
    assert "other" not in candidates or exact_jaccard(set(base), other) > 0.1


def test_lsh_threshold_configuration_and_recall_measurement():
    """Step 40 + section 8.2: thresholds tune bands; recall is measured against exact pairs."""
    config = threshold_config_for(0.8)
    assert 0.6 <= config.threshold() <= 0.95
    signer = MinHashSigner(_schema(), permutations=128)
    config = choose_bands(128, 0.8)
    index = LshBandIndex(config)
    rng = random.Random(101)
    universe = [f"f-{i}" for i in range(300)]
    records = {}
    exact_pairs = set()
    ids = []
    for record_index in range(40):
        features = set(rng.sample(universe, 100))
        record_id = f"rec-{record_index:02d}"
        ids.append(record_id)
        records[record_id] = features
        index.add(signer.signature(record_id, features))
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            if exact_jaccard(records[ids[i]], records[ids[j]]) >= 0.8:
                exact_pairs.add((ids[i], ids[j]))
    report = measure_recall(index, exact_pairs, similarity_threshold=0.8)
    assert report.recall >= 0.5  # LSH trades recall for cost; measured, not assumed
    assert report.to_dict()["candidate_volume"] == index.bucket_count()
    assert report.precision <= 1.0 + 1e-9


# ---------------------------------------------------------------------------
# Step 41


def test_simhash_fingerprints_and_hamming_distance():
    """Step 41: near-identical feature vectors produce near fingerprints."""
    features_a = {"alpha": 2.0, "beta": 1.5, "gamma": -1.0, "delta": 0.5}
    features_b = {"alpha": 2.0, "beta": 1.5, "gamma": -1.0, "epsilon": 0.5}
    features_c = {f"noise-{i}": 1.0 for i in range(50)}
    fingerprint_a = simhash("a", features_a)
    fingerprint_b = simhash("b", features_b)
    fingerprint_c = simhash("c", features_c)
    assert isinstance(fingerprint_a, SimHashFingerprint)
    assert hamming_distance(fingerprint_a, fingerprint_b) <= 8
    assert hamming_distance(fingerprint_a, fingerprint_c) >= 8
    assert validate_distance_threshold(4)
    assert not validate_distance_threshold(48)
    with pytest.raises(ValueError):
        simhash("x", {})
    with pytest.raises(TypeError):
        simhash("x", {"feature": "not-a-number"})  # type: ignore[arg-type]


def test_simhash_index_nominates_candidates_within_distance():
    """Step 41 + 42: chunk bucketing nominates near-duplicate candidates."""
    index = SimHashIndex(bits=64, chunk_bits=8)  # 8 chunks: distance <= 7 guaranteed
    base = {f"feature-{i}": 1.0 for i in range(40)}
    near = dict(base)
    near["feature-40"] = 1.0
    far = {f"noise-{i}": 2.0 for i in range(40)}
    fingerprint_base = simhash("base", base)
    fingerprint_near = simhash("near", near)
    fingerprint_far = simhash("far", far)
    index.add_all([fingerprint_base, fingerprint_near, fingerprint_far])
    distance = hamming_distance(fingerprint_base, fingerprint_near)
    assert distance <= index.chunk_count - 1  # within the guaranteed bucket range
    candidates = index.candidates(fingerprint_base, max_distance=8)
    assert "near" in candidates
    assert "far" not in candidates
    assert index.records() == 3


# ---------------------------------------------------------------------------
# Steps 42-43


def test_candidates_are_verified_exactly_before_any_merge():
    """Step 42: a candidate score alone never merges records."""
    first = {f"a-{i}" for i in range(50)}
    second = {f"a-{i}" for i in range(45)} | {f"b-{i}" for i in range(50)}
    verified = verify_candidate_similarity(
        ("rec-1", "rec-2"), candidate_score=0.7, first_features=first, second_features=second,
        threshold=0.8,
    )
    assert isinstance(verified, VerifiedSimilarity)
    assert verified.verified is False  # exact Jaccard ~ 45/105 < 0.8
    rejected = verify_candidate_similarity(
        ("rec-1", "rec-2"), candidate_score=0.95, first_features=first, second_features=second,
        threshold=0.8,
    )
    assert rejected.verified is False  # even a high candidate score cannot merge
    identical = verify_candidate_similarity(
        ("rec-1", "rec-1b"), candidate_score=0.9, first_features=first, second_features=set(first),
        threshold=0.8,
    )
    assert identical.verified is True
    assert identical.exact_value == 1.0


def test_byte_identity_and_structured_field_verification():
    """Step 42: canonical byte equality and structured field comparison verify candidates."""
    byte_same = verify_byte_identity(("x", "y"), b"same", b"same")
    assert byte_same.verified is True
    byte_diff = verify_byte_identity(("x", "z"), b"same", b"different")
    assert byte_diff.verified is False
    fields_same = verify_structured_fields(
        ("x", "y"), {"kind": "CONCEPT", "ns": "a"}, {"kind": "CONCEPT", "ns": "a"},
        required_equal=("kind", "ns"),
    )
    assert fields_same.verified is True
    fields_diff = verify_structured_fields(
        ("x", "y"), {"kind": "CONCEPT", "ns": "a"}, {"kind": "CONCEPT", "ns": "b"},
        required_equal=("ns",),
    )
    assert fields_diff.verified is False
    assert fields_diff.notes[0].startswith("fields differ")


def test_duplicate_classification_and_provenance_are_recorded():
    """Step 43: byte-identical, near-duplicate, related and distinct stay separate."""
    verified_near = verify_candidate_similarity(
        ("a", "b"), candidate_score=0.85,
        first_features={f"x{i}" for i in range(100)},
        second_features={f"x{i}" for i in range(95)} | {f"y{i}" for i in range(5)},
        threshold=0.6,
    )
    classification = classify_duplicate(
        candidate_score=0.85, verified_similarity=verified_near,
    )
    assert classification is DuplicateClassification.NEAR_DUPLICATE_DOCUMENT
    byte_identical = classify_duplicate(
        candidate_score=1.0,
        verified_similarity=verify_byte_identity(("a", "b"), b"z", b"z"),
        byte_identical=True,
    )
    assert byte_identical is DuplicateClassification.BYTE_IDENTICAL
    related = classify_duplicate(
        candidate_score=0.3,
        verified_similarity=verify_candidate_similarity(
            ("a", "b"), candidate_score=0.3,
            first_features={f"x{i}" for i in range(100)},
            second_features={f"y{i}" for i in range(100)},
            threshold=0.8,
        ),
    )
    assert related is DuplicateClassification.SEMANTICALLY_RELATED
    equivalent = classify_duplicate(
        candidate_score=0.1,
        verified_similarity=verify_candidate_similarity(
            ("a", "b"), candidate_score=0.1, first_features={"x"}, second_features={"x"},
            threshold=0.8,
        ),
        logically_equivalent_ids=True,
    )
    assert equivalent is DuplicateClassification.LOGICALLY_EQUIVALENT
    provenance = record_duplicate_provenance(
        pair=("a", "b"),
        classification=classification,
        feature_schema=_schema().schema_id(),
        thresholds={"similarity_threshold": 0.6},
        candidate_score=0.85,
        exact_score=verified_near.exact_value,
        verification_method=verified_near.verification_method,
        verification_result=verified_near.verified,
    )
    payload = provenance.to_dict()
    assert payload["algorithm"] == "minhash+lsh"
    assert payload["feature_schema"].startswith("token_shingles")
    assert payload["thresholds"]["similarity_threshold"] == 0.6
    assert payload["verification_result"] is True
    assert payload["created_at"]
