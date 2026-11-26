"""Numerical comparison, adversarial fixtures, determinism, downstream
decision stability, and regression benchmark records (Phase G).

Implements the comparison layer every optimized result must pass before
promotion: absolute/relative error statistics with combined tolerance
semantics, adversarial fixtures (extreme logits, near-singular systems,
rank-deficient matrices, overflow-scale inputs), determinism checks,
downstream decision stability under input perturbation, and serializable
regression benchmark records keyed by configuration hash.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .contracts import CorrectnessContract, ErrorReport


def compare_arrays(
    candidate: np.ndarray,
    reference: np.ndarray,
    contract: CorrectnessContract,
    *,
    notes: tuple[str, ...] = (),
) -> ErrorReport:
    """Compare optimized and reference outputs (Phase G comparison)."""
    candidate_values = np.asarray(candidate, dtype=np.float64)
    reference_values = np.asarray(reference, dtype=np.float64)
    if candidate_values.shape != reference_values.shape:
        return ErrorReport(
            within_tolerance=False,
            tolerance_basis="shape mismatch",
            sample_count=0,
            notes=notes + (f"shape mismatch {candidate_values.shape} vs {reference_values.shape}",),
        )
    absolute = np.abs(candidate_values - reference_values)
    denominator = np.maximum(np.abs(reference_values), 1e-30)
    relative = absolute / denominator
    elementwise_ok = absolute <= (
        contract.abs_tolerance + contract.rel_tolerance * np.abs(reference_values)
    )
    max_abs = float(absolute.max()) if absolute.size else 0.0
    return ErrorReport(
        max_abs=max_abs,
        mean_abs=float(absolute.mean()) if absolute.size else 0.0,
        max_rel=float(relative.max()) if relative.size else 0.0,
        mean_rel=float(relative.mean()) if relative.size else 0.0,
        within_tolerance=bool(np.all(elementwise_ok)) if elementwise_ok.size else True,
        tolerance_basis=(
            f"|diff| <= {contract.abs_tolerance:g} + {contract.rel_tolerance:g} * |ref|"
        ),
        sample_count=int(candidate_values.size),
        notes=notes,
    )


def check_attention_properties(
    output: np.ndarray,
    v: np.ndarray,
    mask: np.ndarray | None,
    *,
    conservation_required: bool = True,
) -> dict[str, Any]:
    """Property checks for attention outputs (conservation and value hull)."""
    values = np.asarray(v, dtype=np.float64)
    result = np.asarray(output, dtype=np.float64)
    report: dict[str, Any] = {
        "finite": bool(np.all(np.isfinite(result))),
        "within_value_hull": bool(
            result.min() >= values.min() - 1e-9 and result.max() <= values.max() + 1e-9
        ),
    }
    if conservation_required and mask is not None:
        rows_with_keys = np.any(mask, axis=1)
        report["conservation_supported_rows"] = bool(rows_with_keys.all())
    return report


def determinism_check(
    fn: Callable[[Any], Any], inputs: Any, repeats: int = 2
) -> dict[str, Any]:
    """Verify repeated execution produces identical outputs (contract)."""
    outputs = [fn(inputs) for _ in range(repeats)]
    identical = all(
        np.array_equal(np.asarray(outputs[0]), np.asarray(other)) for other in outputs[1:]
    )
    return {"deterministic": identical, "repeats": repeats}


def downstream_decision_stability(
    fn: Callable[[Any], np.ndarray],
    reference_fn: Callable[[Any], np.ndarray],
    inputs: Any,
    *,
    epsilon: float = 1e-6,
    seed: int = 5,
) -> dict[str, Any]:
    """Check that the decision readout is stable under small input perturbation.

    The decision readout is the argmax of the output; a perturbation of size
    epsilon must not flip it between the optimized and reference paths.
    """
    rng = np.random.default_rng(seed)
    baseline_opt = fn(inputs)
    baseline_ref = reference_fn(inputs)
    perturbed = {
        key: (
            np.asarray(value) + epsilon * rng.standard_normal(np.asarray(value).shape)
            if isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.floating)
            else value
        )
        for key, value in inputs.items()
    }
    perturbed_opt = fn(perturbed)
    perturbed_ref = reference_fn(perturbed)
    decisions = []
    for array in (baseline_opt, baseline_ref, perturbed_opt, perturbed_ref):
        flat = np.asarray(array, dtype=np.float64).ravel()
        decisions.append(int(np.argmax(flat)) if flat.size else -1)
    stable = decisions[0] == decisions[1] and decisions[2] == decisions[3]
    return {
        "stable": bool(stable),
        "decisions": decisions,
        "epsilon": epsilon,
    }


# ---------------------------------------------------------------------------
# Adversarial fixtures (Phase G)
# ---------------------------------------------------------------------------


def adversarial_attention_fixture(sequence: int = 24, dim: int = 8, seed: int = 31) -> dict[str, Any]:
    """Extreme-logit attention with a partial mask including masked rows."""
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((sequence, dim)) * 80.0
    k = rng.standard_normal((sequence, dim)) * 80.0
    v = rng.standard_normal((sequence, dim))
    mask = rng.random((sequence, sequence)) > 0.25
    mask[: sequence // 2] = True
    return {"q": q, "k": k, "v": v, "mask": mask}


def adversarial_least_squares(seed: int = 31) -> dict[str, Any]:
    """Near-singular least squares system (condition ~1e12)."""
    rng = np.random.default_rng(seed)
    rows, columns = 40, 6
    u, _ = np.linalg.qr(rng.standard_normal((rows, columns)))
    v, _ = np.linalg.qr(rng.standard_normal((columns, columns)))
    singular = np.logspace(0, -12, columns)
    A = (u * singular) @ v.T
    b = A @ rng.standard_normal(columns)
    return {"A": A, "b": b, "condition": float(np.linalg.cond(A))}


def adversarial_rank_deficient_matrix(seed: int = 31) -> np.ndarray:
    """Exactly rank-deficient matrix for sketch rank checks."""
    rng = np.random.default_rng(seed)
    left = rng.standard_normal((32, 2))
    right = rng.standard_normal((2, 48))
    return left @ right


def adversarial_overflow_matrix(seed: int = 31) -> np.ndarray:
    """Values whose products overflow float16 (range guard fixture)."""
    rng = np.random.default_rng(seed)
    return rng.uniform(200.0, 400.0, size=(24, 24))


def run_adversarial_suite(
    attention_impl: Callable[[dict[str, Any]], np.ndarray],
    attention_reference: Callable[[dict[str, Any]], np.ndarray],
    *,
    tolerance: float = 1e-8,
) -> dict[str, Any]:
    """Run the adversarial comparison suite for attention implementations."""
    fixture = adversarial_attention_fixture()
    optimized = attention_impl(fixture)
    reference = attention_reference(fixture)
    maximum = float(np.max(np.abs(optimized - reference)))
    return {
        "adversarial_attention_max_abs": maximum,
        "within_tolerance": maximum <= tolerance,
        "finite": bool(np.all(np.isfinite(optimized))),
    }


# ---------------------------------------------------------------------------
# Regression benchmark records (Phase G)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RegressionRecord:
    """Serializable benchmark record keyed by configuration hash."""

    name: str
    config_hash: str
    metrics: dict[str, float]
    fixture_seed: int
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "config_hash": self.config_hash,
            "metrics": dict(self.metrics),
            "fixture_seed": self.fixture_seed,
            "notes": list(self.notes),
        }


def save_regression_records(
    records: tuple[RegressionRecord, ...], path: str | Path
) -> dict[str, Any]:
    """Persist regression records as JSON for reproduction."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {"records": [record.to_dict() for record in records]}
    target.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return {"path": str(target), "record_count": len(records)}


def load_regression_records(path: str | Path) -> list[RegressionRecord]:
    """Load regression records written by save_regression_records."""
    payload = json.loads(Path(path).read_text())
    return [
        RegressionRecord(
            name=row["name"],
            config_hash=row["config_hash"],
            metrics=dict(row["metrics"]),
            fixture_seed=int(row["fixture_seed"]),
            notes=tuple(row["notes"]),
        )
        for row in payload["records"]
    ]
