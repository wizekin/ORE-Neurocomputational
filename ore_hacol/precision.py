"""Mixed-precision numerical policies (integrated steps 37-42).

Implements operation-specific dtype selection (FP16/BF16/FP32/FP64), a CPU
emulation of bfloat16 via round-to-nearest-even mantissa truncation, explicit
product-versus-accumulation precision separation in blocked matrix products,
range/overflow/underflow checks, iterative refinement for eligible linear
systems, higher-precision verification, and rejection of unsafe precision.
Global dtype changes are never applied.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .contracts import CorrectnessContract, ErrorReport

#: Numeric limits of the candidate formats (engineering values; BF16 emulated on CPU).
DTYPE_LIMITS: dict[str, dict[str, float]] = {
    "float16": {"max": 65504.0, "min_normal": 6.104e-5, "epsilon": 9.765625e-4},
    "bfloat16": {"max": 3.3895313892515355e38, "min_normal": 1.175494350822287e-38,
                 "epsilon": 7.8125e-3},
    "float32": {"max": 3.4028234663852886e38, "min_normal": 1.175494350822287e-38,
                "epsilon": 1.1920928955078125e-7},
    "float64": {"max": 1.7976931348623157e308, "min_normal": 2.2250738585072014e-308,
                "epsilon": 2.220446049250313e-16},
}

#: Conservative safety margin on range before a smaller format is accepted.
_RANGE_SAFETY = 4.0


def to_bfloat16(x: np.ndarray) -> np.ndarray:
    """Round float32 values to emulated bfloat16 (8-bit exponent, 7-bit mantissa).

    Round-to-nearest-even is applied to the truncated low 16 mantissa bits;
    NaN and Inf pass through unchanged.
    """
    values = np.asarray(x, dtype=np.float32)
    bits = values.view(np.uint32)
    special = (bits >> 23) == 0xFF  # NaN / Inf exponent field
    rounded = bits + np.uint32(0x7FFF) + ((bits >> 16) & np.uint32(1))
    truncated = ((rounded >> np.uint32(16)) << np.uint32(16)).astype(np.uint32)
    result_bits = np.where(special, bits, truncated)
    return result_bits.view(np.float32)


@dataclass(frozen=True)
class RangeReport:
    """Overflow/underflow audit of values against a candidate dtype (step 39)."""

    dtype_name: str
    finite: bool
    overflow_count: int
    underflow_subnormal_count: int
    max_abs: float
    safe: bool
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "dtype": self.dtype_name,
            "finite": self.finite,
            "overflow_count": self.overflow_count,
            "underflow_subnormal_count": self.underflow_subnormal_count,
            "max_abs": self.max_abs,
            "safe": self.safe,
            "notes": list(self.notes),
        }


def check_range(x: np.ndarray, dtype_name: str) -> RangeReport:
    """Check values against a dtype's representable range (integrated step 39)."""
    values = np.asarray(x, dtype=np.float64)
    limits = DTYPE_LIMITS[dtype_name]
    finite = bool(np.all(np.isfinite(values)))
    magnitude = np.abs(values[np.isfinite(values)]) if values.size else np.zeros(0)
    max_abs = float(magnitude.max()) if magnitude.size else 0.0
    overflow = int(np.sum(magnitude > limits["max"] / _RANGE_SAFETY)) if magnitude.size else 0
    underflow = int(
        np.sum((magnitude > 0.0) & (magnitude < limits["min_normal"]))
    ) if magnitude.size else 0
    safe = finite and overflow == 0
    return RangeReport(
        dtype_name=dtype_name,
        finite=finite,
        overflow_count=overflow,
        underflow_subnormal_count=underflow,
        max_abs=max_abs,
        safe=safe,
        notes=("values are never silently clipped; overflow forces a higher-precision fallback",),
    )


@dataclass(frozen=True)
class PrecisionDecision:
    """Per-operation dtype decision with product and accumulation dtypes (step 37-38)."""

    product_dtype: str
    accumulate_dtype: str
    safe: bool
    rationale: str
    rejected_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "product_dtype": self.product_dtype,
            "accumulate_dtype": self.accumulate_dtype,
            "safe": self.safe,
            "rationale": self.rationale,
            "rejected_reason": self.rejected_reason,
        }


def choose_precision(
    value_max_abs: float,
    condition_estimate: float | None,
    contract: CorrectnessContract,
    accumulation_depth: int = 1,
) -> PrecisionDecision:
    """Select product/accumulation dtypes per operation (integrated step 37).

    Rules: exactness required -> FP64 throughout; dynamic range gates FP16;
    tolerance versus unit roundoff times sqrt(accumulation depth) gates BF16 and
    FP16; conditioning may force FP64 accumulation; unsafe choices are
    rejected with an explicit reason rather than silently applied.
    """
    depth = max(1, int(accumulation_depth))
    sqrt_depth = float(np.sqrt(depth))
    tolerance = max(contract.rel_tolerance, 1e-16)
    condition = condition_estimate or 1.0

    def error_bound(dtype_name: str) -> float:
        return DTYPE_LIMITS[dtype_name]["epsilon"] * sqrt_depth * 10.0 * max(1.0, np.log10(max(condition, 10.0)))

    candidates: list[str] = []
    if value_max_abs > 0 and value_max_abs <= DTYPE_LIMITS["float16"]["max"] / _RANGE_SAFETY:
        candidates.append("float16")
    if value_max_abs <= DTYPE_LIMITS["bfloat16"]["max"] / _RANGE_SAFETY:
        candidates.append("bfloat16")
    if value_max_abs <= DTYPE_LIMITS["float32"]["max"] / _RANGE_SAFETY:
        candidates.append("float32")
    candidates.append("float64")

    if contract.exactness_required:
        return PrecisionDecision(
            product_dtype="float64",
            accumulate_dtype="float64",
            safe=True,
            rationale="exactness required; randomized/low-precision paths disabled",
        )
    for dtype_name in candidates:
        if error_bound(dtype_name) <= tolerance:
            accumulate = "float32" if dtype_name in ("float16", "bfloat16") else dtype_name
            if condition > 1e6 and accumulate != "float64":
                accumulate = "float64"
            return PrecisionDecision(
                product_dtype=dtype_name,
                accumulate_dtype=accumulate,
                safe=True,
                rationale=(
                    f"error bound {error_bound(dtype_name):.2e} within tolerance "
                    f"{tolerance:.2e} at depth {depth} and condition {condition:.2e}"
                ),
            )
    return PrecisionDecision(
        product_dtype="float64",
        accumulate_dtype="float64",
        safe=True,
        rationale="no candidate format satisfied the error budget; FP64 enforced",
        rejected_reason="float16/bfloat16/float32 error bounds exceed the allowed tolerance",
    )


def mixed_precision_matmul(
    A: np.ndarray,
    B: np.ndarray,
    *,
    product_dtype: str = "float16",
    accumulate_dtype: str = "float32",
    block: int = 64,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Blocked matrix product with separated product and accumulation precision.

    Products are evaluated in the lower-precision product dtype and each
    block's contribution is accumulated in the higher-precision accumulator,
    implementing integrated step 38. The report records actual behavior
    including any overflow observed in the low-precision products.
    """
    left = np.asarray(A, dtype=np.float64)
    right = np.asarray(B, dtype=np.float64)
    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[0]:
        raise ValueError("A must be (m, k) and B must be (k, n)")
    m, k = left.shape
    n = right.shape[1]
    if product_dtype == "bfloat16":
        cast = lambda values: to_bfloat16(values)  # noqa: E731 - local alias
    else:
        cast = lambda values: values.astype(product_dtype)  # noqa: E731

    accumulator = np.zeros((m, n), dtype=accumulate_dtype)
    overflow_blocks = 0
    blocks = 0
    for i0 in range(0, m, block):
        for j0 in range(0, n, block):
            for k0 in range(0, k, block):
                with np.errstate(over="ignore", invalid="ignore"):
                    partial = cast(left[i0 : i0 + block, k0 : k0 + block]) @ cast(
                        right[k0 : k0 + block, j0 : j0 + block]
                    )
                    if not np.all(np.isfinite(partial)) or np.any(
                        np.abs(partial)
                        > DTYPE_LIMITS.get(product_dtype, DTYPE_LIMITS["float64"])["max"]
                    ):
                        overflow_blocks += 1
                    accumulator[i0 : i0 + block, j0 : j0 + block] += partial.astype(
                        accumulate_dtype
                    )
                blocks += 1
    report = {
        "product_dtype": product_dtype,
        "accumulate_dtype": accumulate_dtype,
        "blocks": blocks,
        "overflow_blocks": overflow_blocks,
        "notes": [
            "products evaluated in the product dtype; blocks accumulated in the accumulate dtype",
        ],
    }
    return accumulator.astype(np.float64), report


def iterative_refinement(
    A: np.ndarray,
    b: np.ndarray,
    *,
    solve_dtype: str = "float32",
    residual_dtype: str = "float64",
    max_iterations: int = 4,
    tolerance: float = 1e-10,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Mixed-precision iterative refinement (integrated step 40).

    Solves at lower precision, computes the residual at higher precision,
    solves the correction at lower precision, and updates until the
    convergence test passes; conditioning is checked first.
    """
    matrix = np.asarray(A, dtype=np.float64)
    target = np.asarray(b, dtype=np.float64)
    condition = float(np.linalg.cond(matrix))
    solve_matrix = matrix.astype(solve_dtype)

    def _solve_low(rhs: np.ndarray) -> np.ndarray:
        # Least-squares solve at the declared (lower) precision; works for
        # square and rectangular systems alike.
        return np.linalg.lstsq(solve_matrix, rhs.astype(solve_dtype), rcond=None)[0].astype(
            np.float64
        )

    x = _solve_low(target)
    target_norm = float(np.linalg.norm(target))
    initial_residual = float(np.linalg.norm(matrix @ x - target))
    history: list[dict[str, float]] = []
    converged = initial_residual <= tolerance * max(target_norm, 1e-300)
    iterations = 0
    while not converged and iterations < max_iterations:
        residual = (target - matrix @ x).astype(residual_dtype)
        correction = _solve_low(residual)
        x = x + correction
        iterations += 1
        current = float(np.linalg.norm(matrix @ x - target))
        history.append({"iteration": iterations, "residual": current})
        converged = current <= tolerance * max(target_norm, 1e-300)
    final_residual = float(np.linalg.norm(matrix @ x - target))
    report = {
        "condition_estimate": condition,
        "initial_residual": initial_residual,
        "final_residual": final_residual,
        "iterations": iterations,
        "converged": bool(converged),
        "residual_history": history,
        "solve_dtype": solve_dtype,
        "residual_dtype": residual_dtype,
        "eligible": bool(condition < 1e12),
        "notes": [
            "residuals computed at higher precision; corrections solved at lower precision",
            "conditioning checked before refinement is trusted",
        ],
    }
    return x, report


def verify_high_precision(
    candidate: np.ndarray,
    reference: np.ndarray,
    contract: CorrectnessContract,
) -> ErrorReport:
    """Recompute comparison against an FP64 reference (integrated step 41).

    Acceptance follows the standard combined test |diff| <= atol + rtol |ref|;
    elementwise relative error is reported for diagnostics but never used
    alone when reference values approach zero.
    """
    candidate_values = np.asarray(candidate, dtype=np.float64)
    reference_values = np.asarray(reference, dtype=np.float64)
    absolute = np.abs(candidate_values - reference_values)
    denominator = np.maximum(np.abs(reference_values), 1e-30)
    relative = absolute / denominator
    elementwise_ok = absolute <= contract.abs_tolerance + contract.rel_tolerance * np.abs(
        reference_values
    )
    max_abs = float(absolute.max()) if absolute.size else 0.0
    max_rel = float(relative.max()) if relative.size else 0.0
    within = bool(np.all(elementwise_ok)) if elementwise_ok.size else True
    return ErrorReport(
        max_abs=max_abs,
        mean_abs=float(absolute.mean()) if absolute.size else 0.0,
        max_rel=max_rel,
        mean_rel=float(relative.mean()) if relative.size else 0.0,
        within_tolerance=within,
        tolerance_basis=(
            f"|diff| <= {contract.abs_tolerance:g} + {contract.rel_tolerance:g} * |ref| (fp64 reference)"
        ),
        sample_count=int(candidate_values.size),
        notes=("higher-precision verification detects injected low-precision errors",),
    )


def reject_unsafe_precision(
    error_report: ErrorReport,
    range_report: RangeReport | None,
    condition_estimate: float | None,
    *,
    condition_limit: float = 1e12,
) -> tuple[bool, str]:
    """Decide whether a precision plan must be rejected (integrated step 42)."""
    if not error_report.within_tolerance:
        return True, f"measured error exceeds the contract (max_rel={error_report.max_rel:.3e})"
    if range_report is not None and not range_report.safe:
        return True, f"range violations in {range_report.dtype_name} (overflow={range_report.overflow_count})"
    if condition_estimate is not None and condition_estimate > condition_limit:
        return True, f"condition estimate {condition_estimate:.3e} exceeds limit {condition_limit:g}"
    return False, "precision plan accepted"
