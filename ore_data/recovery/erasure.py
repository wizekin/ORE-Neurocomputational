"""Reed-Solomon erasure-coded archives over GF(2^8) (Phase J, Step 56).

Erasure coding exists only for selected archives or unreliable distribution
channels: data shards k and total shards n are declared with failure
assumptions and a reconstruction threshold; any k of n shards reconstruct
the content, and reconstruction verifies every shard hash and the final
content hash before returning data (source section 4.8 and 8.3). Parity
overhead is (n-k)/k relative to the k data shards. Erasure coding is never
a replacement for backups, access control or cryptographic integrity
(source 1.2).

The code is a systematic Reed-Solomon over GF(2^8) with the primitive
polynomial x^8 + x^4 + x^3 + x^2 + 1 (0x11D): the generator matrix is a
Vandermonde matrix whose top k x k block is normalized to the identity, so
shard i < k equals data shard i and shards k..n-1 are parity.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Sequence, Tuple


#: Primitive polynomial of GF(2^8): x^8+x^4+x^3+x^2+x^0 -> 0x11D.
FIELD_POLYNOMIAL = 0x11D
#: Multiplicative order of the field's multiplicative group.
FIELD_SIZE = 256
#: Maximum supported shard count (field-constructible Vandermonde nodes).
MAX_SHARDS = 255


class ErasureCodingError(RuntimeError):
    """Raised on shard corruption or impossible reconstruction."""


# --------------------------------------------------------------------- GF(256)

def _build_tables() -> tuple[list[int], list[int]]:
    """Build the GF(2^8) exponent and logarithm tables."""
    exp = [0] * 512
    log = [0] * FIELD_SIZE
    value = 1
    for power in range(255):
        exp[power] = value
        log[value] = power
        value <<= 1
        if value & 0x100:
            value ^= FIELD_POLYNOMIAL
    for power in range(255, 512):
        exp[power] = exp[power - 255]
    return exp, log


_EXP, _LOG = _build_tables()


def gf_mul(a: int, b: int) -> int:
    """Multiply two field elements."""
    if a == 0 or b == 0:
        return 0
    return _EXP[_LOG[a] + _LOG[b]]


def gf_div(a: int, b: int) -> int:
    """Divide two field elements (b must be nonzero)."""
    if b == 0:
        raise ZeroDivisionError("GF division by zero")
    if a == 0:
        return 0
    return _EXP[(_LOG[a] - _LOG[b]) % 255]


def gf_pow(a: int, power: int) -> int:
    """Raise one field element to an integer power."""
    result = 1
    base = a
    exponent = power
    while exponent:
        if exponent & 1:
            result = gf_mul(result, base)
        base = gf_mul(base, base)
        exponent >>= 1
    return result


def gf_inv(a: int) -> int:
    """Invert one nonzero field element."""
    if a == 0:
        raise ZeroDivisionError("cannot invert field zero")
    return _EXP[255 - _LOG[a]]


# --------------------------------------------------------------------- matrices

def _matrix_multiply(matrix: list[list[int]], vector: list[int]) -> list[int]:
    """Multiply one GF matrix by one column vector."""
    return [
        _dot_product(row, vector) for row in matrix
    ]


def _dot_product(row: list[int], vector: list[int]) -> int:
    """Inner product over GF(2^8)."""
    total = 0
    for a, b in zip(row, vector):
        total ^= gf_mul(a, b)
    return total


def _matrix_invert(matrix: list[list[int]]) -> list[list[int]]:
    """Invert one k x k GF(2^8) matrix with partial pivoting."""
    size = len(matrix)
    work = [row[:] + [1 if i == j else 0 for j in range(size)] for i, row in enumerate(matrix)]
    for column in range(size):
        pivot_row = next(
            (r for r in range(column, size) if work[r][column] != 0),
            None,
        )
        if pivot_row is None:
            raise ErasureCodingError("matrix is singular; shard set cannot decode")
        work[column], work[pivot_row] = work[pivot_row], work[column]
        pivot = work[column][column]
        inverse = gf_inv(pivot)
        work[column] = [gf_mul(value, inverse) for value in work[column]]
        for r in range(size):
            if r != column and work[r][column] != 0:
                factor = work[r][column]
                work[r] = [
                    work[r][c] ^ gf_mul(factor, work[column][c])
                    for c in range(2 * size)
                ]
    return [row[size:] for row in work]


def _vandermonde(rows: int, columns: int) -> list[list[int]]:
    """Vandermonde matrix with distinct evaluation points alpha_i = 2^i."""
    matrix = []
    for row in range(rows):
        alpha = gf_pow(2, row)
        matrix.append([gf_pow(alpha, column) for column in range(columns)])
    return matrix


def build_generator_matrix(n: int, k: int) -> list[list[int]]:
    """Build the systematic generator: top k x k is the identity (Step 56)."""
    if not 0 < k < n <= MAX_SHARDS:
        raise ErasureCodingError(f"need 0 < k < n <= {MAX_SHARDS}")
    vandermonde = _vandermonde(n, k)
    top = [row[:] for row in vandermonde[:k]]
    top_inverse = _matrix_invert(top)
    generator = []
    for row in vandermonde:
        # G[i] = top_inverse * V[i] (row-vector times matrix) over GF(2^8).
        generated = []
        for column in range(k):
            accumulator = 0
            for inner in range(k):
                accumulator ^= gf_mul(top_inverse[inner][column], row[inner])
            generated.append(accumulator)
        generator.append(generated)
    # Normalize away float-free exact identity check on the top block.
    for i in range(k):
        for j in range(k):
            expected = 1 if i == j else 0
            if generator[i][j] != expected:  # pragma: no cover - math guarantee
                raise ErasureCodingError("generator top block is not the identity")
    return generator


# --------------------------------------------------------------------- archive

@dataclass(frozen=True, slots=True)
class ErasurePolicy:
    """Shard policy of one erasure-coded archive (Step 56)."""

    data_shards: int
    total_shards: int
    failure_assumption: str
    schema_version: str = "0.1.0"

    def __post_init__(self) -> None:
        """Validate the policy."""
        if not 0 < self.data_shards < self.total_shards <= MAX_SHARDS:
            raise ErasureCodingError(
                f"need 0 < data < total <= {MAX_SHARDS}, got "
                f"{self.data_shards}/{self.total_shards}"
            )
        if not self.failure_assumption.strip():
            raise ValueError("failure_assumption must be documented")

    @property
    def parity_shards(self) -> int:
        """Return the number of parity shards."""
        return self.total_shards - self.data_shards

    @property
    def reconstruction_threshold(self) -> int:
        """Return the minimum shard count for reconstruction: k (Step 56)."""
        return self.data_shards

    @property
    def overhead(self) -> float:
        """Return parity overhead (n-k)/k (source 4.8)."""
        return self.parity_shards / self.data_shards

    def to_dict(self) -> dict[str, object]:
        """Return the policy as a plain dictionary."""
        return {
            "data_shards": self.data_shards,
            "parity_shards": self.parity_shards,
            "total_shards": self.total_shards,
            "reconstruction_threshold": self.reconstruction_threshold,
            "overhead": self.overhead,
            "failure_assumption": self.failure_assumption,
            "schema_version": self.schema_version,
        }


@dataclass(frozen=True, slots=True)
class ErasureShard:
    """One encoded shard with its integrity hash (Step 56)."""

    index: int
    data: bytes
    shard_hash: str

    def __post_init__(self) -> None:
        """Validate the shard."""
        if self.index < 0:
            raise ValueError("shard index must be non-negative")
        if not isinstance(self.data, (bytes, bytearray)):
            raise TypeError("shard data must be bytes")

    def verify(self) -> bool:
        """Recompute this shard's hash (source 8.3 shard integrity check)."""
        return hashlib.sha256(bytes(self.data)).hexdigest() == self.shard_hash


@dataclass(frozen=True, slots=True)
class ReconstructedArchive:
    """Result of one reconstruction with verified integrity (Step 56)."""

    content: bytes
    content_id_verified: bool
    shards_used: Tuple[int, ...]
    reconstruction_ms: float
    algorithm: str = "reed_solomon_gf256"
    algorithm_version: str = "0.1.0"

    def __post_init__(self) -> None:
        """Freeze the reconstructed content."""
        object.__setattr__(self, "content", bytes(self.content))

    def to_dict(self) -> dict[str, object]:
        """Return metadata for the reconstruction report."""
        return {
            "bytes": len(self.content),
            "content_id_verified": self.content_id_verified,
            "shards_used": list(self.shards_used),
            "reconstruction_ms": self.reconstruction_ms,
            "algorithm": self.algorithm,
            "algorithm_version": self.algorithm_version,
        }


def _now() -> str:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class ErasureArchive:
    """One erasure-coded archive over GF(2^8) with hash-verified shards (Step 56)."""

    def __init__(self, policy: ErasurePolicy) -> None:
        """Create an empty archive for one shard policy."""
        self.policy = policy
        self._generator = build_generator_matrix(policy.total_shards, policy.data_shards)
        self._content: bytes | None = None
        self._content_length: int = 0
        self._content_digest: str = ""
        self._shards: dict[int, ErasureShard] = {}
        self.created_at: str = ""
        self.reconstructions = 0
        self.failed_reconstructions = 0

    # -- encoding ----------------------------------------------------------

    def encode(self, content: bytes) -> Tuple[ErasureShard, ...]:
        """Encode one content object into n hash-verified shards (Step 56).

        The content is padded to a multiple of k shard blocks, split into
        k data shards, and the parity shards are the generator matrix rows
        applied per byte column.
        """
        if not isinstance(content, (bytes, bytearray)):
            raise TypeError("content must be bytes")
        payload = bytes(content)
        k = self.policy.data_shards
        n = self.policy.total_shards
        block = (len(payload) + k - 1) // k
        padded = payload + b"\x00" * (block * k - len(payload))
        rows = [padded[index * block : (index + 1) * block] for index in range(k)]
        shards: list[ErasureShard] = []
        for index in range(n):
            if index < k:
                shard_bytes = rows[index]
            else:
                parity = bytearray(block)
                for row, coefficient in enumerate(self._generator[index]):
                    if coefficient == 0:
                        continue
                    for position in range(block):
                        parity[position] ^= gf_mul(coefficient, rows[row][position])
                shard_bytes = bytes(parity)
            shards.append(
                ErasureShard(
                    index=index,
                    data=shard_bytes,
                    shard_hash=hashlib.sha256(shard_bytes).hexdigest(),
                )
            )
        self._content = payload
        self._content_length = len(payload)
        self._content_digest = hashlib.sha256(payload).hexdigest()
        self._shards = {shard.index: shard for shard in shards}
        self.created_at = _now()
        return tuple(shards)

    # -- access --------------------------------------------------------------

    def shards(self) -> Tuple[ErasureShard, ...]:
        """Return every shard of the archive."""
        return tuple(self._shards[index] for index in sorted(self._shards))

    def content_digest(self) -> str:
        """Return the digest of the encoded content."""
        if not self._content_digest:
            raise ErasureCodingError("archive is empty; encode first (Step 56)")
        return self._content_digest

    def archive_metadata(self) -> dict[str, object]:
        """Export the reconstruction metadata stored beside the shards (Step 56)."""
        return {
            "content_length": self._content_length,
            "content_digest": self._content_digest,
            "created_at": self.created_at,
            "policy": self.policy.to_dict(),
        }

    @classmethod
    def from_metadata(cls, policy: ErasurePolicy, metadata: dict[str, object]) -> "ErasureArchive":
        """Rebuild one archive shell from stored metadata for standalone reconstruction."""
        archive = cls(policy)
        archive._content_length = int(metadata["content_length"])
        archive._content_digest = str(metadata["content_digest"])
        archive.created_at = str(metadata.get("created_at", ""))
        return archive

    # -- reconstruction ----------------------------------------------------------

    def reconstruct(self, available: Sequence[ErasureShard]) -> ReconstructedArchive:
        """Reconstruct the content from any k valid shards (Step 56).

        Every supplied shard is hash-verified before use; at least k valid
        shards are required; the decode matrix is the generator rows of the
        chosen shards, inverted over GF(2^8); the reconstructed content must
        match the archived content digest and length before it is returned
        (source 8.3).
        """
        started = time.perf_counter()
        k = self.policy.data_shards
        valid: dict[int, bytes] = {}
        for shard in available:
            if not shard.verify():
                raise ErasureCodingError(
                    f"shard {shard.index} failed its integrity hash; corrupt "
                    "shards cannot participate in reconstruction (Step 56)"
                )
            valid[shard.index] = bytes(shard.data)
        if len(valid) < k:
            raise ErasureCodingError(
                f"need at least {k} valid shards, got {len(valid)} (Step 56)"
            )
        chosen = sorted(valid)[:k]
        decode_matrix = [self._generator[index][:k] for index in chosen]
        inverse = _matrix_invert(decode_matrix)
        block = len(valid[chosen[0]])
        recovered = bytearray()
        for row in range(k):
            # data_row = inverse[row] applied over the chosen shard columns.
            row_bytes = bytearray(block)
            for column, shard_index in enumerate(chosen):
                coefficient = inverse[row][column]
                if coefficient == 0:
                    continue
                shard_bytes = valid[shard_index]
                for position in range(block):
                    row_bytes[position] ^= gf_mul(coefficient, shard_bytes[position])
            recovered += bytes(row_bytes)
        content = bytes(recovered[: self._content_length]) if self._content_length else bytes(recovered)
        digest_verified = True
        if self._content_digest:
            digest_verified = hashlib.sha256(content).hexdigest() == self._content_digest
            if not digest_verified:
                self.failed_reconstructions += 1
                raise ErasureCodingError(
                    "reconstructed content failed its content-hash verification "
                    "(Step 56; source 8.3)"
                )
        self.reconstructions += 1
        duration_ms = (time.perf_counter() - started) * 1000.0
        return ReconstructedArchive(
            content=content,
            content_id_verified=digest_verified,
            shards_used=tuple(chosen),
            reconstruction_ms=round(duration_ms, 3),
        )

    # -- reporting -------------------------------------------------------------------

    def describe(self) -> dict[str, object]:
        """Return archive metadata for observability."""
        return {
            "algorithm": "reed_solomon_gf256",
            "version": "0.1.0",
            "policy": self.policy.to_dict(),
            "created_at": self.created_at,
            "shards": len(self._shards),
            "content_digest": self._content_digest,
            "reconstructions": self.reconstructions,
            "failed_reconstructions": self.failed_reconstructions,
        }


def reconstruct_from_any_k(
    policy: ErasurePolicy,
    content: bytes,
    dropped: Sequence[int],
) -> ReconstructedArchive:
    """Acceptance helper: encode, drop declared shards, reconstruct from the rest."""
    archive = ErasureArchive(policy)
    shards = archive.encode(content)
    available = tuple(shard for shard in shards if shard.index not in set(dropped))
    return archive.reconstruct(available)


__all__ = [
    "FIELD_POLYNOMIAL",
    "MAX_SHARDS",
    "ErasureCodingError",
    "gf_mul",
    "gf_div",
    "gf_pow",
    "gf_inv",
    "build_generator_matrix",
    "ErasurePolicy",
    "ErasureShard",
    "ReconstructedArchive",
    "ErasureArchive",
    "reconstruct_from_any_k",
]
