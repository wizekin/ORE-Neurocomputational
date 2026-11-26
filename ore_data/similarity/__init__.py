"""Similarity and duplicate candidate detection (Phase G).

MinHash signatures estimate Jaccard similarity, LSH band buckets generate
candidates, SimHash fingerprints compare by Hamming distance; every
candidate is verified exactly or semantically before any merge, deletion
or canonicalization decision, and duplicate detection preserves full
provenance (Steps 37-43).
"""
from ore_data.similarity.minhash import (
    DEFAULT_PERMUTATIONS,
    DuplicateClassification,
    DuplicateProvenanceRecord,
    FeatureRepresentation,
    FeatureSchema,
    MinHashSignature,
    MinHashSigner,
    SimilarityEstimate,
    VerifiedSimilarity,
    build_feature_set,
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
from ore_data.similarity.lsh import (
    LshBandConfig,
    LshBandIndex,
    LshRecallReport,
    candidate_probability,
    choose_bands,
    measure_recall,
    threshold_config_for,
)
from ore_data.similarity.simhash import (
    DEFAULT_BITS,
    SimHashCandidate,
    SimHashFingerprint,
    SimHashIndex,
    estimated_angular_distance,
    hamming_distance,
    simhash,
    validate_distance_threshold,
)

__all__ = [
    "DEFAULT_PERMUTATIONS",
    "DuplicateClassification",
    "DuplicateProvenanceRecord",
    "FeatureRepresentation",
    "FeatureSchema",
    "MinHashSignature",
    "MinHashSigner",
    "SimilarityEstimate",
    "VerifiedSimilarity",
    "build_feature_set",
    "build_shingles",
    "classify_duplicate",
    "estimated_jaccard",
    "exact_jaccard",
    "normalize_tokens",
    "record_duplicate_provenance",
    "verify_byte_identity",
    "verify_candidate_similarity",
    "verify_structured_fields",
    "LshBandConfig",
    "LshBandIndex",
    "LshRecallReport",
    "candidate_probability",
    "choose_bands",
    "measure_recall",
    "threshold_config_for",
    "DEFAULT_BITS",
    "SimHashCandidate",
    "SimHashFingerprint",
    "SimHashIndex",
    "estimated_angular_distance",
    "hamming_distance",
    "simhash",
    "validate_distance_threshold",
]
