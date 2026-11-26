"""MinHash similarity signatures with exact verification and provenance (Phase G).

Steps 37-39, 42-43: the feature representation being compared is declared
explicitly (Step 37); canonical feature sets preserve language, tokenizer,
normalization and feature-schema versions so signatures stay comparable
(Step 38); MinHash signatures estimate Jaccard similarity with r hash
components whose matching fraction is the estimator and whose variance is
approximately J(1-J)/r (Step 39, source 4.4). Candidate pairs are always
verified with exact identifiers, canonical byte equality or structured
comparison before any merge or deletion decision (Step 42), and duplicate
detection keeps full provenance: algorithm, schema, thresholds, score,
verification result and duplicate classification (Step 43).
"""
from __future__ import annotations

import hashlib
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence, Set


class FeatureRepresentation(str, Enum):
    """The representation being compared (Step 37)."""

    TOKEN_SHINGLES = "token_shingles"
    NORMALIZED_TEXT_TOKENS = "normalized_text_tokens"
    SEMANTIC_FEATURES = "semantic_features"
    GRAPH_NEIGHBORHOODS = "graph_neighborhoods"
    ARTIFACT_BYTES = "artifact_bytes"


#: Default MinHash permutation count: estimator variance ~ J(1-J)/r.
DEFAULT_PERMUTATIONS = 128


@dataclass(frozen=True, slots=True)
class FeatureSchema:
    """Canonical feature schema making similarity signatures comparable (Step 38)."""

    representation: FeatureRepresentation
    shingle_width: int
    language: str
    tokenizer_version: str
    normalization_version: str
    schema_version: str = "0.1.0"

    def __post_init__(self) -> None:
        """Validate the schema descriptor."""
        if not isinstance(self.representation, FeatureRepresentation):
            raise TypeError("representation must be a FeatureRepresentation")
        if self.shingle_width <= 0:
            raise ValueError("shingle_width must be positive")
        if not self.language.strip():
            raise ValueError("language must be non-empty")

    def schema_id(self) -> str:
        """Return the stable schema identity framing every signature."""
        return (
            f"{self.representation.value}|k={self.shingle_width}|lang={self.language}"
            f"|tok={self.tokenizer_version}|norm={self.normalization_version}"
            f"|v={self.schema_version}"
        )

    def to_dict(self) -> dict[str, str | int]:
        """Return the schema descriptor as a plain dictionary."""
        return {
            "representation": self.representation.value,
            "shingle_width": self.shingle_width,
            "language": self.language,
            "tokenizer_version": self.tokenizer_version,
            "normalization_version": self.normalization_version,
            "schema_version": self.schema_version,
        }


@dataclass(frozen=True, slots=True)
class SimilarityEstimate:
    """Typed MinHash Jaccard estimate: a candidate score, not truth (Step 39)."""

    value: float
    permutations: int
    variance_note: str
    feature_schema: str
    algorithm: str = "minhash"
    algorithm_version: str = "0.1.0"
    timestamp: str = ""
    source_watermark: int = 0

    def __post_init__(self) -> None:
        """Validate and stamp the estimate."""
        if not 0.0 <= self.value <= 1.0:
            raise ValueError("similarity values must be within [0, 1]")
        if not self.timestamp:
            object.__setattr__(self, "timestamp", _now())

    @property
    def exact(self) -> bool:
        """MinHash similarity is an estimate unless verified exactly (Step 42)."""
        return False

    def to_dict(self) -> dict[str, object]:
        """Return the estimate with its full metadata (section 5.1 requirement)."""
        return {
            "algorithm": self.algorithm,
            "algorithm_version": self.algorithm_version,
            "value": self.value,
            "permutations": self.permutations,
            "variance_note": self.variance_note,
            "feature_schema": self.feature_schema,
            "timestamp": self.timestamp,
            "source_watermark": self.source_watermark,
            "exact": self.exact,
        }


@dataclass(frozen=True, slots=True)
class VerifiedSimilarity:
    """Result of exact or semantic verification of one candidate (Step 42)."""

    pair: tuple[str, str]
    verification_method: str
    verified: bool
    exact_value: float | None
    candidate_score: float | None
    notes: tuple[str, ...] = ()
    algorithm: str = "exact_verification"
    algorithm_version: str = "0.1.0"

    def to_dict(self) -> dict[str, object]:
        """Return the verification result as a plain dictionary."""
        return {
            "pair": list(self.pair),
            "verification_method": self.verification_method,
            "verified": self.verified,
            "exact_value": self.exact_value,
            "candidate_score": self.candidate_score,
            "notes": list(self.notes),
            "algorithm": self.algorithm,
            "algorithm_version": self.algorithm_version,
        }


class DuplicateClassification(str, Enum):
    """Duplicate classifications kept distinct in provenance (Step 43)."""

    BYTE_IDENTICAL = "byte_identical"
    NEAR_DUPLICATE_DOCUMENT = "near_duplicate_document"
    SEMANTICALLY_RELATED = "semantically_related"
    LOGICALLY_EQUIVALENT = "logically_equivalent"
    DISTINCT = "distinct"


@dataclass(frozen=True, slots=True)
class DuplicateProvenanceRecord:
    """Provenance record of one duplicate-detection decision (Step 43)."""

    pair: tuple[str, str]
    classification: DuplicateClassification
    algorithm: str
    algorithm_version: str
    feature_schema: str
    thresholds: Mapping[str, float]
    candidate_score: float | None
    exact_score: float | None
    verification_method: str
    verification_result: bool
    reviewer: str = "ore_data.similarity"
    created_at: str = ""

    def __post_init__(self) -> None:
        """Freeze and stamp the provenance record."""
        object.__setattr__(self, "thresholds", dict(self.thresholds))
        if not self.created_at:
            object.__setattr__(self, "created_at", _now())

    def to_dict(self) -> dict[str, object]:
        """Return the provenance record as a plain dictionary."""
        return {
            "pair": list(self.pair),
            "classification": self.classification.value,
            "algorithm": self.algorithm,
            "algorithm_version": self.algorithm_version,
            "feature_schema": self.feature_schema,
            "thresholds": dict(self.thresholds),
            "candidate_score": self.candidate_score,
            "exact_score": self.exact_score,
            "verification_method": self.verification_method,
            "verification_result": self.verification_result,
            "reviewer": self.reviewer,
            "created_at": self.created_at,
        }


def _now() -> str:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def normalize_tokens(text: str) -> tuple[str, ...]:
    """Normalize text into canonical tokens (Step 38).

    NFKC normalization, case folding and whitespace tokenization; the
    normalization version in the feature schema records this pipeline.
    """
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return tuple(token for token in normalized.split() if token)


def build_shingles(text: str, schema: FeatureSchema) -> frozenset[str]:
    """Construct the canonical token shingle set of one text (Step 38)."""
    tokens = normalize_tokens(text)
    width = schema.shingle_width
    if len(tokens) < width:
        return frozenset({(" ".join(tokens))} if tokens else set())
    return frozenset(
        " ".join(tokens[index : index + width])
        for index in range(len(tokens) - width + 1)
    )


def build_feature_set(
    features: Iterable[str],
    schema: FeatureSchema,
) -> frozenset[str]:
    """Canonicalize an explicit semantic feature identifier set (Step 38)."""
    return frozenset(str(feature) for feature in features)


def _component_hash(schema_id: str, component: int, feature: str) -> int:
    """One permutation hash component: h_i(x) over the framed feature (Step 39)."""
    digest = hashlib.sha256(f"minhash|{schema_id}|{component}|{feature}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


@dataclass(frozen=True, slots=True)
class MinHashSignature:
    """One set's MinHash signature under one feature schema (Step 39)."""

    record_id: str
    values: tuple[int, ...]
    feature_count: int
    schema: FeatureSchema
    algorithm: str = "minhash"
    algorithm_version: str = "0.1.0"
    source_watermark: int = 0

    def __post_init__(self) -> None:
        """Validate the signature."""
        if not self.values:
            raise ValueError("signatures require at least one permutation")
        object.__setattr__(self, "values", tuple(self.values))
        if not self.record_id.strip():
            raise ValueError("record_id must be non-empty")

    def band(self, band_index: int, rows: int) -> tuple[int, ...]:
        """Return one band slice of the signature for LSH bucketing (Step 40)."""
        start = band_index * rows
        band = self.values[start : start + rows]
        if len(band) != rows:
            raise ValueError(f"band {band_index} exceeds signature length")
        return band

    def to_dict(self) -> dict[str, object]:
        """Return the signature metadata as a plain dictionary."""
        return {
            "record_id": self.record_id,
            "permutations": len(self.values),
            "feature_count": self.feature_count,
            "feature_schema": self.schema.schema_id(),
            "algorithm": self.algorithm,
            "algorithm_version": self.algorithm_version,
            "source_watermark": self.source_watermark,
        }


class MinHashSigner:
    """Generate MinHash signatures under one canonical feature schema (Step 39)."""

    def __init__(
        self,
        schema: FeatureSchema,
        *,
        permutations: int = DEFAULT_PERMUTATIONS,
    ) -> None:
        """Create one signer for one schema and permutation count."""
        if permutations < 2:
            raise ValueError("permutations must be at least 2")
        self.schema = schema
        self.permutations = permutations

    def signature(
        self,
        record_id: str,
        features: Iterable[str],
        *,
        watermark: int = 0,
    ) -> MinHashSignature:
        """Compute sig(A)_i = min h_i(x) over the feature set (Step 39)."""
        feature_set = frozenset(str(feature) for feature in features)
        if not feature_set:
            raise ValueError("cannot sign an empty feature set")
        schema_id = self.schema.schema_id()
        values = tuple(
            min(_component_hash(schema_id, component, feature) for feature in feature_set)
            for component in range(self.permutations)
        )
        return MinHashSignature(
            record_id=record_id,
            values=values,
            feature_count=len(feature_set),
            schema=self.schema,
            source_watermark=watermark,
        )

    def signature_from_text(
        self,
        record_id: str,
        text: str,
        *,
        watermark: int = 0,
    ) -> MinHashSignature:
        """Sign one text's canonical shingle set (Step 38)."""
        return self.signature(
            record_id,
            build_shingles(text, self.schema),
            watermark=watermark,
        )


def estimated_jaccard(
    first: MinHashSignature,
    second: MinHashSignature,
) -> SimilarityEstimate:
    """Estimate Jaccard similarity as the fraction of matching components (Step 39)."""
    if first.schema.schema_id() != second.schema.schema_id():
        raise ValueError(
            "cannot compare signatures from different feature schemas; "
            "similarity signatures stay comparable only under one schema (Step 38)"
        )
    if len(first.values) != len(second.values):
        raise ValueError("cannot compare signatures of different lengths")
    matched = sum(1 for a, b in zip(first.values, second.values) if a == b)
    r = len(first.values)
    return SimilarityEstimate(
        value=matched / r,
        permutations=r,
        variance_note="estimator variance is approximately J(1-J)/r (source 4.4)",
        feature_schema=first.schema.schema_id(),
        timestamp=_now(),
        source_watermark=max(first.source_watermark, second.source_watermark),
    )


def exact_jaccard(first: Set[str], second: Set[str]) -> float:
    """Compute the exact Jaccard similarity of two feature sets (Step 42)."""
    if not first and not second:
        return 1.0
    union = first | second
    if not union:
        return 1.0
    return len(first & second) / len(union)


def verify_candidate_similarity(
    candidate_pair: tuple[str, str],
    *,
    candidate_score: float | None,
    first_features: Set[str],
    second_features: Set[str],
    threshold: float = 0.8,
) -> VerifiedSimilarity:
    """Verify one candidate pair with exact set similarity (Step 42).

    Verification is required before merging, deleting or canonicalizing
    records: a candidate score alone never justifies any of those actions.
    """
    exact = exact_jaccard(first_features, second_features)
    verified = exact >= threshold
    return VerifiedSimilarity(
        pair=candidate_pair,
        verification_method="exact_set_similarity",
        verified=verified,
        exact_value=exact,
        candidate_score=candidate_score,
        notes=(
            (
                "candidate verified by exact set similarity; never merge solely "
                "because an LSH bucket suggested similarity"
            )
            if verified
            else ("candidate rejected: exact similarity below threshold",)
        ),
    )


def verify_byte_identity(
    candidate_pair: tuple[str, str],
    first_bytes: bytes,
    second_bytes: bytes,
) -> VerifiedSimilarity:
    """Verify one candidate pair by canonical byte equality (Step 42)."""
    verified = first_bytes == second_bytes
    return VerifiedSimilarity(
        pair=candidate_pair,
        verification_method="canonical_byte_equality",
        verified=verified,
        exact_value=1.0 if verified else 0.0,
        candidate_score=None,
        notes=(
            ("byte-identical duplicates confirmed",) if verified
            else ("byte sequences differ; not a byte-identical duplicate",)
        ),
    )


def verify_structured_fields(
    candidate_pair: tuple[str, str],
    first_fields: Mapping[str, Any],
    second_fields: Mapping[str, Any],
    *,
    required_equal: Sequence[str] = (),
) -> VerifiedSimilarity:
    """Verify one candidate pair by structured field comparison (Step 42)."""
    fields = tuple(required_equal) or tuple(sorted(set(first_fields) & set(second_fields)))
    mismatches = tuple(
        field for field in fields if first_fields.get(field) != second_fields.get(field)
    )
    verified = not mismatches
    return VerifiedSimilarity(
        pair=candidate_pair,
        verification_method="structured_field_comparison",
        verified=verified,
        exact_value=None,
        candidate_score=None,
        notes=() if verified else (f"fields differ: {', '.join(mismatches)}",),
    )


def classify_duplicate(
    *,
    candidate_score: float | None,
    verified_similarity: VerifiedSimilarity,
    byte_identical: bool = False,
    logically_equivalent_ids: bool = False,
    near_duplicate_threshold: float = 0.8,
) -> DuplicateClassification:
    """Classify one verified pair into the duplicate taxonomy (Step 43)."""
    if byte_identical:
        return DuplicateClassification.BYTE_IDENTICAL
    if logically_equivalent_ids:
        return DuplicateClassification.LOGICALLY_EQUIVALENT
    if verified_similarity.verified and (
        verified_similarity.exact_value is not None
        and verified_similarity.exact_value >= near_duplicate_threshold
    ):
        return DuplicateClassification.NEAR_DUPLICATE_DOCUMENT
    if candidate_score is not None and candidate_score > 0.0:
        return DuplicateClassification.SEMANTICALLY_RELATED
    return DuplicateClassification.DISTINCT


def record_duplicate_provenance(
    *,
    pair: tuple[str, str],
    classification: DuplicateClassification,
    feature_schema: str,
    thresholds: Mapping[str, float],
    candidate_score: float | None,
    exact_score: float | None,
    verification_method: str,
    verification_result: bool,
    reviewer: str = "ore_data.similarity",
) -> DuplicateProvenanceRecord:
    """Record full duplicate-detection provenance (Step 43)."""
    return DuplicateProvenanceRecord(
        pair=pair,
        classification=classification,
        algorithm="minhash+lsh",
        algorithm_version="0.1.0",
        feature_schema=feature_schema,
        thresholds=dict(thresholds),
        candidate_score=candidate_score,
        exact_score=exact_score,
        verification_method=verification_method,
        verification_result=verification_result,
        reviewer=reviewer,
        created_at=_now(),
    )


__all__ = [
    "FeatureRepresentation",
    "DEFAULT_PERMUTATIONS",
    "FeatureSchema",
    "SimilarityEstimate",
    "VerifiedSimilarity",
    "DuplicateClassification",
    "DuplicateProvenanceRecord",
    "MinHashSignature",
    "MinHashSigner",
    "normalize_tokens",
    "build_shingles",
    "build_feature_set",
    "estimated_jaccard",
    "exact_jaccard",
    "verify_candidate_similarity",
    "verify_byte_identity",
    "verify_structured_fields",
    "classify_duplicate",
    "record_duplicate_provenance",
]
