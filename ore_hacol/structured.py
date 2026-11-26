"""Structured matrix and transform representations (integrated steps 33-36).

Implements butterfly factorizations (O(n log n) apply versus O(n^2) dense),
Toeplitz operators via FFT convolution, structure detection tests
(butterfly/Toeplitz-like/sparsity/low-rank-block), verification against dense
references, and preprocessing-amortization measurement. Structure is only
exploited when reconstruction against the dense reference verifies within
tolerance -- a hoped-for speedup is not evidence.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import scipy.linalg
import scipy.signal


@dataclass(frozen=True)
class StructureReport:
    """Structure detection evidence (integrated step 33)."""

    toeplitz_like: bool = False
    butterfly_certificate: bool = False
    sparsity_fraction: float = 1.0
    low_rank_block_fraction: float = 0.0
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "toeplitz_like": self.toeplitz_like,
            "butterfly_certificate": self.butterfly_certificate,
            "sparsity_fraction": self.sparsity_fraction,
            "low_rank_block_fraction": self.low_rank_block_fraction,
            "notes": list(self.notes),
        }


def _require_power_of_two(size: int) -> int:
    if size < 2 or size & (size - 1):
        raise ValueError(f"size must be a power of two >= 2, got {size}")
    return int(size)


class ButterflyFactors:
    """Butterfly factorization B = B_L ... B_1 with 2x2 blocks per stage.

    Each stage is a block-diagonal matrix of 2x2 blocks acting on index pairs
    (t, t + 2^(s-1)) within groups of size 2^s; applying all stages to a
    vector costs O(n log n) arithmetic versus O(n^2) for the dense product.
    """

    def __init__(self, blocks: tuple[np.ndarray, ...], size: int, seed: int) -> None:
        self.blocks = blocks
        self.size = size
        self.seed = seed
        self.stages = len(blocks)

    @classmethod
    def build(cls, size: int, seed: int = 21) -> "ButterflyFactors":
        """Construct seeded butterfly factors with a documented distribution."""
        _require_power_of_two(size)
        rng = np.random.default_rng(seed)
        stages = int(np.log2(size))
        blocks = tuple(rng.standard_normal((size // 2, 2, 2)) for _ in range(stages))
        return cls(blocks, size, seed)

    def apply(self, x: np.ndarray) -> np.ndarray:
        """Apply the butterfly to a vector or matrix column-by-column (O(n log n))."""
        vector = np.asarray(x, dtype=np.float64)
        if vector.ndim == 1:
            if vector.shape[0] != self.size:
                raise ValueError("input length must equal matrix size")
            current = vector.copy()
            for stage, blocks in enumerate(self.blocks, start=1):
                current = _apply_stage(current, blocks, stage)
            return current
        if vector.ndim == 2 and vector.shape[0] == self.size:
            return np.stack([self.apply(column) for column in vector.T], axis=1)
        raise ValueError("input must be a length-n vector or an n-by-k matrix")

    def to_dense(self) -> np.ndarray:
        """Materialize the dense product for verification/reference paths."""
        result = np.eye(self.size, dtype=np.float64)
        for column in range(self.size):
            unit = np.zeros(self.size)
            unit[column] = 1.0
            result[:, column] = self.apply(unit)
        return result

    def describe(self) -> dict[str, Any]:
        return {
            "size": self.size,
            "stages": self.stages,
            "seed": self.seed,
            "apply_flops": 2 * self.size * self.stages,
            "dense_flops": 2 * self.size**2,
        }


def _apply_stage(x: np.ndarray, blocks: np.ndarray, stage: int) -> np.ndarray:
    """Apply one block-diagonal stage of 2x2 blocks to a vector."""
    group = 1 << stage
    half = group >> 1
    reshaped = x.reshape(-1, group)
    left = reshaped[:, :half]
    right = reshaped[:, half:]
    pair_blocks = blocks.reshape(reshaped.shape[0], half, 2, 2)
    a = pair_blocks[..., 0, 0]
    b = pair_blocks[..., 0, 1]
    c = pair_blocks[..., 1, 0]
    d = pair_blocks[..., 1, 1]
    y_left = a * left + b * right
    y_right = c * left + d * right
    return np.concatenate((y_left, y_right), axis=1).reshape(-1)


def butterfly_certificate(A: np.ndarray, factors: ButterflyFactors, tolerance: float = 1e-10) -> bool:
    """Verify that the factors reproduce the dense matrix within tolerance."""
    matrix = np.asarray(A, dtype=np.float64)
    if matrix.shape != (factors.size, factors.size):
        return False
    dense = factors.to_dense()
    return bool(np.allclose(matrix, dense, atol=tolerance, rtol=tolerance))


@dataclass(frozen=True)
class ToeplitzOperator:
    """Toeplitz operator with first column and first row."""

    first_column: np.ndarray
    first_row: np.ndarray

    @classmethod
    def build(cls, first_column: np.ndarray, first_row: np.ndarray | None = None) -> "ToeplitzOperator":
        column = np.asarray(first_column, dtype=np.float64).ravel()
        row = np.asarray(first_row if first_row is not None else column, dtype=np.float64).ravel()
        if row[0] != column[0]:
            raise ValueError("first_row[0] must equal first_column[0]")
        return cls(column, row)

    def to_dense(self) -> np.ndarray:
        return scipy.linalg.toeplitz(self.first_column, self.first_row)

    def matvec(self, x: np.ndarray) -> np.ndarray:
        """Fast apply via FFT convolution (O(n log n)) with a dense reference available.

        y[i] = sum_{j<=i} c[i-j] x[j] + sum_{j>i} r[j-i] x[j]; the upper part is
        computed by convolving with the reversed first row and removing the
        diagonal term r[0] x[i] that the correlation slice would double-count.
        """
        vector = np.asarray(x, dtype=np.float64).ravel()
        size = vector.shape[0]
        if size != self.first_column.shape[0]:
            raise ValueError("input length must equal operator size")
        lower = scipy.signal.fftconvolve(vector, self.first_column)[:size]
        reversed_row = self.first_row[::-1]
        upper_full = scipy.signal.fftconvolve(vector, reversed_row)
        # upper[i] = sum_{j>=i} r[j-i] x[j]; subtract the diagonal r[0] x[i].
        upper = upper_full[size - 1 : 2 * size - 1] - vector * self.first_row[0]
        return lower + upper

    def describe(self) -> dict[str, Any]:
        size = self.first_column.shape[0]
        return {
            "size": size,
            "apply_flops_fft": int(size * np.log2(max(2, size)) * 6),
            "dense_flops": 2 * size**2,
        }


def detect_structure(
    A: np.ndarray,
    *,
    toeplitz_tolerance: float = 1e-8,
    sparsity_threshold: float = 1e-12,
    block: int = 8,
    low_rank_tolerance: float = 1e-8,
) -> StructureReport:
    """Test for exploitable structure (integrated step 33).

    Butterfly structure requires an explicit factor certificate; Toeplitz
    likeness requires constant diagonals; sparsity and low-rank blocks are
    measured fractions. Nothing is claimed without demonstrated evidence.
    """
    matrix = np.asarray(A, dtype=np.float64)
    scale = max(float(np.max(np.abs(matrix))), 1e-300)
    # Toeplitz-like: every diagonal is (nearly) constant.
    toeplitz_like = True
    for offset in range(-(matrix.shape[0] - 1), matrix.shape[1]):
        diagonal = np.diagonal(matrix, offset=offset)
        if diagonal.size > 1 and diagonal.max() - diagonal.min() > toeplitz_tolerance * scale:
            toeplitz_like = False
            break
    sparsity_fraction = float(np.mean(np.abs(matrix) > sparsity_threshold))
    rows, columns = matrix.shape
    block = min(block, rows, columns)
    block_rows = rows // block
    block_columns = columns // block
    low_rank_blocks = 0
    total_blocks = block_rows * block_columns
    if total_blocks:
        for i in range(block_rows):
            for j in range(block_columns):
                piece = matrix[i * block : (i + 1) * block, j * block : (j + 1) * block]
                singular = np.linalg.svd(piece, compute_uv=False)
                energy = float(np.sum(singular**2))
                if energy == 0.0:
                    low_rank_blocks += 1
                    continue
                trailing = float(np.sum(singular[1:] ** 2))
                if trailing <= low_rank_tolerance**2 * energy:
                    low_rank_blocks += 1
    return StructureReport(
        toeplitz_like=toeplitz_like,
        sparsity_fraction=sparsity_fraction,
        low_rank_block_fraction=(low_rank_blocks / total_blocks) if total_blocks else 0.0,
        notes=(
            "butterfly exploitation requires an explicit verified factor certificate",
            "detection reports evidence only; verification against dense reference is mandatory",
        ),
    )


@dataclass(frozen=True)
class VerificationReport:
    """Structured-versus-dense verification evidence (integrated step 36)."""

    max_abs_error: float
    mean_abs_error: float
    within_tolerance: bool
    sample_count: int
    adversarial_max_abs_error: float
    rejected: bool
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_abs_error": self.max_abs_error,
            "mean_abs_error": self.mean_abs_error,
            "within_tolerance": self.within_tolerance,
            "sample_count": self.sample_count,
            "adversarial_max_abs_error": self.adversarial_max_abs_error,
            "rejected": self.rejected,
            "notes": list(self.notes),
        }


def verify_structured_apply(
    dense_apply: Callable[[np.ndarray], np.ndarray],
    structured_apply: Callable[[np.ndarray], np.ndarray],
    inputs: tuple[np.ndarray, ...],
    tolerance: float = 1e-9,
    *,
    adversarial_inputs: tuple[np.ndarray, ...] = (),
) -> VerificationReport:
    """Compare structured and dense outputs on representative and adversarial inputs."""
    errors = []
    for sample in inputs:
        errors.append(float(np.max(np.abs(structured_apply(sample) - dense_apply(sample)))))
    adversarial_errors = [
        float(np.max(np.abs(structured_apply(sample) - dense_apply(sample))))
        for sample in adversarial_inputs
    ]
    worst_adversarial = max(adversarial_errors, default=0.0)
    max_error = max(errors, default=0.0)
    within = max_error <= tolerance and worst_adversarial <= tolerance
    rejected = not within
    return VerificationReport(
        max_abs_error=max_error,
        mean_abs_error=float(np.mean(errors)) if errors else 0.0,
        within_tolerance=within,
        sample_count=len(inputs),
        adversarial_max_abs_error=worst_adversarial,
        rejected=rejected,
        notes=("reject when accuracy or runtime is worse than the dense reference",),
    )


@dataclass(frozen=True)
class AmortizationReport:
    """Preprocessing amortization measurement (integrated step 35)."""

    build_seconds: float
    structured_seconds_per_apply: float
    dense_seconds_per_apply: float
    applications: int
    break_even_applications: float
    net_benefit: bool
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "build_seconds": self.build_seconds,
            "structured_seconds_per_apply": self.structured_seconds_per_apply,
            "dense_seconds_per_apply": self.dense_seconds_per_apply,
            "applications": self.applications,
            "break_even_applications": self.break_even_applications,
            "net_benefit": self.net_benefit,
            "notes": list(self.notes),
        }


def measure_amortization(
    build_callable: Callable[[], Any],
    apply_callable: Callable[[np.ndarray], np.ndarray],
    dense_callable: Callable[[np.ndarray], np.ndarray],
    inputs: tuple[np.ndarray, ...],
    applications: int | None = None,
) -> AmortizationReport:
    """Measure preprocessing amortization over repeated applications (step 35)."""
    start = time.perf_counter()
    build_callable()
    build_seconds = time.perf_counter() - start
    structured_times = []
    dense_times = []
    for sample in inputs:
        start = time.perf_counter()
        apply_callable(sample)
        structured_times.append(time.perf_counter() - start)
        start = time.perf_counter()
        dense_callable(sample)
        dense_times.append(time.perf_counter() - start)
    structured_avg = float(np.mean(structured_times)) if structured_times else 0.0
    dense_avg = float(np.mean(dense_times)) if dense_times else 0.0
    count = applications or len(inputs)
    per_application_gain = dense_avg - structured_avg
    break_even = (build_seconds / per_application_gain) if per_application_gain > 0 else float("inf")
    net_benefit = bool(per_application_gain > 0 and count > break_even)
    return AmortizationReport(
        build_seconds=build_seconds,
        structured_seconds_per_apply=structured_avg,
        dense_seconds_per_apply=dense_avg,
        applications=count,
        break_even_applications=float(break_even) if break_even != float("inf") else float("inf"),
        net_benefit=net_benefit,
        notes=("measure preprocessing amortization; reject if runtime is worse",),
    )
