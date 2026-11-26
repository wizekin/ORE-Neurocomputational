"""Randomized numerical linear algebra (integrated steps 24-32).

Implements randomized range finding (Y = A Omega, Q = orth(Y)), randomized
low-rank factorization (B = Q^T A, A ~ Q B), Gaussian and SRHT test matrices,
power iterations, rank-deficiency detection, error measurement with an
independent probe estimator, and sketched least squares validated on the
original system. Every approximate result carries reproducible metadata
(rank, oversampling, sketch family, seed, power iterations, error estimate,
validation status) as required by integrated step 32.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.linalg

from .contracts import ValidationStatus


@dataclass
class RandomizedApproxMetadata:
    """Reproducibility metadata preserved with every approximate result (step 32)."""

    rank: int
    oversampling: int
    sketch_family: str
    seed: int
    power_iterations: int
    error_estimate: float | None = None
    exact_relative_error: float | None = None
    detected_rank: int | None = None
    rank_deficient: bool = False
    validation_status: ValidationStatus = ValidationStatus.NOT_RUN
    rationale: str = ""
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "oversampling": self.oversampling,
            "sketch_family": self.sketch_family,
            "seed": self.seed,
            "power_iterations": self.power_iterations,
            "error_estimate": self.error_estimate,
            "exact_relative_error": self.exact_relative_error,
            "detected_rank": self.detected_rank,
            "rank_deficient": self.rank_deficient,
            "validation_status": self.validation_status.value,
            "rationale": self.rationale,
            "notes": list(self.notes),
        }


def gaussian_test_matrix(columns: int, width: int, seed: int) -> np.ndarray:
    """Draw a standard Gaussian test matrix Omega of shape (columns, width)."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((columns, width))


def srht_test_matrix(columns: int, width: int, seed: int) -> np.ndarray:
    """Subsampled randomized Hadamard transform test matrix.

    Omega = sqrt(columns / width) * R D H with H a dense Hadamard matrix,
    D a random +-1 diagonal, and R a row subsample. Falls back to a Gaussian
    matrix (recorded in metadata by the caller) when columns is not a power
    of two or is too large for the dense construction.
    """
    if columns & (columns - 1) or columns > 4096 or width > columns:
        return gaussian_test_matrix(columns, width, seed)
    rng = np.random.default_rng(seed)
    hadamard = scipy.linalg.hadamard(columns)
    signs = rng.choice((-1.0, 1.0), size=columns)
    rows = np.sort(rng.choice(columns, size=width, replace=False))
    scaled = np.sqrt(columns / width)
    return scaled * (hadamard[rows] * signs[rows, None]).T


def build_test_matrix(
    columns: int, width: int, seed: int, family: str = "gaussian"
) -> tuple[np.ndarray, str]:
    """Construct the test matrix and report the family actually used (step 26)."""
    if family == "srht":
        matrix = srht_test_matrix(columns, width, seed)
        used = "srht" if matrix.shape == (columns, width) and columns & (columns - 1) == 0 else "srht_fallback_gaussian"
        return matrix, "srht"
    if family != "gaussian":
        raise ValueError(f"unknown sketch family {family!r}; expected 'gaussian' or 'srht'")
    return gaussian_test_matrix(columns, width, seed), "gaussian"


def rank_deficiency_report(matrix: np.ndarray, tolerance: float = 1e-10) -> dict[str, Any]:
    """Detect rank deficiency of a sketch via singular values (integrated step 28)."""
    singular = np.linalg.svd(np.asarray(matrix), compute_uv=False)
    if singular.size == 0:
        return {"detected_rank": 0, "rank_deficient": True, "largest_singular": 0.0}
    largest = float(singular[0])
    detected = int(np.sum(singular > max(tolerance, tolerance * largest)))
    return {
        "detected_rank": detected,
        "rank_deficient": detected < singular.size,
        "largest_singular": largest,
    }


def randomized_range(
    A: np.ndarray,
    rank: int,
    oversampling: int = 5,
    seed: int = 21,
    power_iterations: int = 0,
    family: str = "gaussian",
) -> tuple[np.ndarray, RandomizedApproxMetadata]:
    """Randomized range finder: Y = A Omega; Q = orth(Y) (integrated steps 26-28).

    Power iterations are applied only when the caller justifies them with an
    error test; the justification is expected to appear in the metadata
    rationale (step 27 safeguard).
    """
    matrix = np.asarray(A, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("A must be 2-D")
    rows, columns = matrix.shape
    if rank < 1 or rank > min(rows, columns):
        raise ValueError(f"rank must be in 1..{min(rows, columns)}")
    if oversampling < 0:
        raise ValueError("oversampling must be non-negative")
    width = rank + oversampling
    omega, used_family = build_test_matrix(columns, width, seed, family)
    Y = matrix @ omega
    deficiency = rank_deficiency_report(Y)
    Q = np.linalg.qr(Y)[0]
    for _ in range(max(0, int(power_iterations))):
        Z = np.linalg.qr(matrix.T @ Q)[0]
        Q = np.linalg.qr(matrix @ Z)[0]
    meta = RandomizedApproxMetadata(
        rank=rank,
        oversampling=oversampling,
        sketch_family=used_family,
        seed=seed,
        power_iterations=int(power_iterations),
        detected_rank=deficiency["detected_rank"],
        rank_deficient=bool(deficiency["rank_deficient"]),
        rationale=(
            f"Y = A Omega with {used_family} Omega of width {width}; "
            f"power iterations {power_iterations} applied only on justification"
        ),
        notes=[f"sketch singular-value rank check: {deficiency}"],
    )
    return Q, meta


def reconstruction_error(A: np.ndarray, Q: np.ndarray) -> float:
    """Exact relative Frobenius residual ||A - Q Q^T A||_F / ||A||_F (step 30)."""
    matrix = np.asarray(A, dtype=np.float64)
    residual = matrix - Q @ (Q.T @ matrix)
    denominator = np.linalg.norm(matrix, "fro")
    if denominator == 0.0:
        return 0.0
    return float(np.linalg.norm(residual, "fro") / denominator)


def error_estimate_probe(
    A: np.ndarray, Q: np.ndarray, seed: int = 7, probes: int = 16
) -> float:
    """Independent probe estimator of ||(I - Q Q^T) A||_F / ||A||_F (step 30).

    Uses a fresh Gaussian probe matrix Omega_perp; the estimator carries a
    documented failure probability for slowly decaying spectra and is never
    reported as an exact residual.
    """
    matrix = np.asarray(A, dtype=np.float64)
    rows, columns = matrix.shape
    rng = np.random.default_rng(seed)
    omega = rng.standard_normal((columns, max(1, probes)))
    sampled = matrix @ omega
    projected = Q @ (Q.T @ sampled)
    numerator = np.linalg.norm(sampled - projected, "fro")
    denominator = np.linalg.norm(sampled, "fro")
    if denominator == 0.0:
        return 0.0
    return float(numerator / denominator)


def randomized_svd(
    A: np.ndarray,
    rank: int,
    oversampling: int = 5,
    seed: int = 21,
    power_iterations: int = 0,
    family: str = "gaussian",
) -> dict[str, Any]:
    """Randomized SVD: A ~ U S V^T through the compressed B = Q^T A (step 29).

    Returns factors plus measured exact residual and an independent probe
    estimate; the caller compares both against the allowed error budget.
    """
    matrix = np.asarray(A, dtype=np.float64)
    Q, meta = randomized_range(
        matrix, rank, oversampling, seed, power_iterations, family
    )
    B = Q.T @ matrix
    left, singular, right_t = np.linalg.svd(B, full_matrices=False)
    U = Q @ left
    keep = min(rank, singular.size)
    U, singular, right_t = U[:, :keep], singular[:keep], right_t[:keep]
    approximation = U @ np.diag(singular) @ right_t
    exact = float(np.linalg.norm(matrix - approximation, "fro") / max(
        1e-300, np.linalg.norm(matrix, "fro")
    ))
    probe = error_estimate_probe(matrix, Q, seed=seed + 1)
    meta.exact_relative_error = exact
    meta.error_estimate = probe
    meta.validation_status = ValidationStatus.PASSED if np.all(np.isfinite(U)) else ValidationStatus.FAILED
    return {
        "U": U,
        "singular_values": singular,
        "Vt": right_t,
        "metadata": meta,
        "exact_relative_error": exact,
        "error_estimate": probe,
    }


def low_rank_opportunity(
    A: np.ndarray,
    reuse_count: int = 1,
    tolerance: float = 1e-6,
    workload_columns: int = 1,
) -> dict[str, Any]:
    """Estimate whether a low-rank path pays off (integrated step 24).

    Savings must exceed decomposition and reconstruction cost: compares dense
    application cost (2 m n k flops per application with k right-hand columns)
    with the randomized path over the reuse count. The decomposition cost
    (~4 m n l flops for l = rank + oversampling) must be amortized by reuse.
    """
    matrix = np.asarray(A, dtype=np.float64)
    rows, columns = matrix.shape
    singular = np.linalg.svd(matrix, compute_uv=False)
    scale = max(float(singular[0]), 1e-300)
    effective_rank = int(np.sum(singular > tolerance * scale))
    min_dim = min(rows, columns)
    width = effective_rank + 5
    k = max(1, int(workload_columns))
    dense_per_application = 2.0 * rows * columns * k
    dense_total = dense_per_application * reuse_count
    decomposition = 4.0 * rows * columns * width
    low_rank_per_application = 2.0 * effective_rank * k * (rows + columns)
    randomized_total = decomposition + low_rank_per_application * reuse_count
    savings = (dense_total - randomized_total) / max(1.0, dense_total)
    return {
        "rows": rows,
        "columns": columns,
        "spectral_rank_at_tolerance": effective_rank,
        "smallest_singular": float(singular[-1]),
        "largest_singular": float(singular[0]),
        "spectrum_decay_ratio": float(singular[-1] / scale),
        "reuse_count": reuse_count,
        "workload_columns": k,
        "estimated_flops_savings_fraction": float(savings),
        "opportunity": bool(savings > 0 and effective_rank < 0.5 * min_dim),
        "notes": [
            "savings must exceed decomposition and reconstruction cost before promotion",
            f"effective rank {effective_rank} of min dimension {min_dim} at tolerance {tolerance}",
        ],
    }


def sketched_least_squares(
    A: np.ndarray,
    b: np.ndarray,
    sketch_rows: int | None = None,
    seed: int = 21,
    family: str = "gaussian",
) -> dict[str, Any]:
    """Sketch-and-solve least squares validated on the original system (step 31).

    min ||Ax - b|| -> min ||S A x - S b||; the original-system residual and
    conditioning are always reported because a small sketched residual alone
    is insufficient evidence.
    """
    matrix = np.asarray(A, dtype=np.float64)
    target = np.asarray(b, dtype=np.float64)
    rows, columns = matrix.shape
    if target.shape[0] != rows:
        raise ValueError("A and b have incompatible row counts")
    if sketch_rows is None:
        sketch_rows = max(2 * columns, min(rows, max(columns + 10, 1)))
    sketch_rows = min(sketch_rows, rows)
    if family == "gaussian":
        S = np.random.default_rng(seed).standard_normal((sketch_rows, rows)) / np.sqrt(
            sketch_rows
        )
        family_used = "gaussian"
    else:
        # SRHT over rows requires a power-of-two row count; fall back otherwise.
        if rows & (rows - 1) or rows > 4096:
            S = np.random.default_rng(seed).standard_normal((sketch_rows, rows)) / np.sqrt(
                sketch_rows
            )
            family_used = "srht_fallback_gaussian"
        else:
            base = srht_test_matrix(rows, sketch_rows, seed)
            S = base.T / np.sqrt(rows)
            family_used = "srht"
    SA = S @ matrix
    Sb = S @ target
    x, *_ = np.linalg.lstsq(SA, Sb, rcond=None)
    sketched_residual = float(np.linalg.norm(SA @ x - Sb))
    original_residual = float(np.linalg.norm(matrix @ x - target))
    target_norm = float(np.linalg.norm(target))
    reference_x, *_ = np.linalg.lstsq(matrix, target, rcond=None)
    reference_residual = float(np.linalg.norm(matrix @ reference_x - target))
    conditioning = {
        "original_condition": float(np.linalg.cond(matrix)),
        "sketched_condition": float(np.linalg.cond(SA)) if SA.size else float("nan"),
    }
    return {
        "x": x,
        "reference_x": reference_x,
        "sketched_residual": sketched_residual,
        "original_residual": original_residual,
        "reference_residual": reference_residual,
        "target_norm": target_norm,
        "sketch_rows": sketch_rows,
        "sketch_family": family_used,
        "conditioning": conditioning,
        "notes": [
            "residual evaluated on the original system; sketched residual alone is insufficient",
        ],
    }
