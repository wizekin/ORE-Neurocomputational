"""Scientific uncertainty, calibration, sensitivity, and out-of-distribution analysis.

Implements Steps 39-42 of the integrated algorithm and the mathematics of Sections
5.6 (ensemble statistics) and 5.7 (sensitivity analysis). Uncertainty kinds are
reported with the method that supports them: ensembles yield aleatoric spread,
cross-method and cross-resolution variation yields epistemic spread, and unsupported
distinctions are listed as limitations rather than being fabricated. Calibration
evidence (Brier score, reliability, interval coverage, sharpness) distinguishes
forecasts that are accurate on average but poorly calibrated. Sensitivity uses the
Jacobian J_ij = dy_i/dtheta_j by finite differences; correlated or ill-conditioned
parameter blocks are flagged non-identifiable so that false precision is never
reported.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from .contracts import UncertaintyReport

__all__ = [
    "UncertaintyEstimator",
    "identifiability_flags",
    "out_of_domain_check",
    "sensitivity_analysis",
]


def sensitivity_analysis(
    function: Callable[[np.ndarray], np.ndarray],
    parameters: np.ndarray,
    *,
    step: float = 1.0e-6,
) -> dict[str, Any]:
    """Compute the normalized local sensitivity matrix J_ij = dy_i/dtheta_j.

    Central finite differences with a relative step; sensitivities are normalized by
    the output scale so the matrix is comparable across outputs.
    """
    parameters = np.asarray(parameters, dtype=float)
    base = np.asarray(function(parameters), dtype=float)
    jacobian = np.zeros((base.shape[0], parameters.shape[0]))
    for column in range(parameters.shape[0]):
        forward_parameters = parameters.copy()
        backward_parameters = parameters.copy()
        magnitude = max(abs(parameters[column]), 1.0)
        forward_parameters[column] += step * magnitude
        backward_parameters[column] -= step * magnitude
        forward = np.asarray(function(forward_parameters), dtype=float)
        backward = np.asarray(function(backward_parameters), dtype=float)
        jacobian[:, column] = (forward - backward) / (
            2.0 * step * magnitude
        )
    scale = max(float(np.max(np.abs(base))), 1.0e-12)
    normalized = jacobian / scale
    return {
        "base_output": base,
        "jacobian": jacobian,
        "normalized_jacobian": normalized,
        "column_norms": np.linalg.norm(normalized, axis=0),
        "method": "central finite differences (Section 5.7)",
    }


def identifiability_flags(
    normalized_jacobian: np.ndarray,
    parameter_names: Sequence[str],
    *,
    condition_threshold: float = 50.0,
    correlation_threshold: float = 0.95,
) -> dict[str, Any]:
    """Flag parameters that cannot be uniquely inferred from the sensitivity matrix.

    Non-identifiability is detected from near-zero sensitivity columns (insensitive
    parameters), near-unit column correlations (redundant parameters), and an
    ill-conditioned Gram matrix. Flagged parameters must not be reported with false
    precision.
    """
    matrix = np.asarray(normalized_jacobian, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != len(parameter_names):
        raise ValueError("jacobian columns must match the parameter names")
    column_norms = np.linalg.norm(matrix, axis=0)
    insensitive = [
        name for name, norm in zip(parameter_names, column_norms) if norm < 1.0e-3
    ]
    redundant: list[tuple[str, str, float]] = []
    for index_a in range(matrix.shape[1]):
        for index_b in range(index_a + 1, matrix.shape[1]):
            column_a, column_b = matrix[:, index_a], matrix[:, index_b]
            norms = np.linalg.norm(column_a) * np.linalg.norm(column_b)
            if norms < 1.0e-18:
                continue
            correlation = float(abs(np.dot(column_a, column_b)) / norms)
            if correlation >= correlation_threshold:
                redundant.append(
                    (parameter_names[index_a], parameter_names[index_b], round(correlation, 4))
                )
    gram = matrix.T @ matrix
    eigenvalues = np.linalg.eigvalsh(gram) if gram.shape[0] > 0 else np.array([1.0])
    positive = eigenvalues[eigenvalues > 1.0e-12]
    condition = float(positive[-1] / positive[0]) if positive.size else float("inf")
    ill_conditioned = condition > condition_threshold ** 2
    return {
        "insensitive_parameters": insensitive,
        "correlated_parameter_pairs": [
            {"a": a, "b": b, "abs_correlation": c} for a, b, c in redundant
        ],
        "gram_condition_number": condition,
        "ill_conditioned": bool(ill_conditioned),
        "non_identifiable": bool(insensitive or redundant or ill_conditioned),
        "guidance": (
            "flagged parameters cannot be uniquely inferred from available data; "
            "do not report them with false precision"
        ),
    }


def out_of_domain_check(
    input_spec: Mapping[str, Any],
    validated_domain: Mapping[str, tuple[float, float]],
) -> dict[str, Any]:
    """Compare input ranges against the model's validated domain (Step 41)."""
    warnings: list[str] = []
    in_domain = True
    for name, value in input_spec.items():
        if name not in validated_domain:
            in_domain = False
            warnings.append(
                f"input {name!r} has no validated domain entry; novelty signal, not proof"
            )
            continue
        low, high = validated_domain[name]
        numeric = float(value)
        if not low <= numeric <= high:
            in_domain = False
            warnings.append(
                f"input {name}={numeric:g} outside validated range [{low:g}, {high:g}]"
            )
    return {
        "in_domain": in_domain,
        "warnings": warnings,
        "action": (
            "conservative solver or expert review"
            if not in_domain
            else "proceed within validated domain"
        ),
    }


class UncertaintyEstimator:
    """Build typed uncertainty reports from ensembles and cross-method evidence."""

    # -- Step 39 ---------------------------------------------------------------
    def from_resolution_ensemble(
        self, probe_values: Sequence[float], *, quantity_name: str
    ) -> UncertaintyReport:
        """Epistemic discretization uncertainty from a resolution ensemble."""
        values = [float(value) for value in probe_values]
        spread = max(values) - min(values) if values else 0.0
        midpoint = values[-1] if values else None
        return UncertaintyReport(
            method="resolution_ensemble",
            point_estimate=midpoint,
            interval=(midpoint - spread, midpoint + spread) if midpoint is not None else None,
            interval_confidence=None,
            epistemic={
                "kind": "discretization/solver variation",
                "spread": spread,
                "levels": values,
            },
            limitations=[
                "interval is a spread across resolutions, not a calibrated confidence interval",
            ],
        )

    def from_cross_method(
        self, estimates: Mapping[str, float], *, quantity_name: str
    ) -> UncertaintyReport:
        """Epistemic disagreement between independent methods (Step 38 companion)."""
        values = [float(value) for value in estimates.values()]
        mean_value = float(np.mean(values))
        spread = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        return UncertaintyReport(
            method="cross_method_disagreement",
            point_estimate=mean_value,
            interval=(mean_value - spread, mean_value + spread),
            interval_confidence=None,
            epistemic={
                "kind": "method disagreement",
                "methods": dict(estimates),
                "spread": spread,
                "shared_assumption_disclosure": (
                    "independence is limited where methods share discretizations or equations"
                ),
            },
            limitations=[
                "genuinely independent validators are preferred; shared assumptions are disclosed",
            ],
        )

    def from_ensemble_forecast(
        self, members: np.ndarray, lead_index: int, *, quantity_name: str
    ) -> UncertaintyReport:
        """Aleatoric forecast uncertainty from ensemble members (Section 5.6)."""
        slice_values = np.asarray(members[:, lead_index], dtype=float)
        mean_value = float(np.mean(slice_values))
        variance = float(np.var(slice_values, ddof=1))
        std = math.sqrt(variance)
        return UncertaintyReport(
            method="ensemble_members",
            point_estimate=mean_value,
            interval=(float(np.quantile(slice_values, 0.05)), float(np.quantile(slice_values, 0.95))),
            interval_confidence=0.90,
            aleatoric={
                "kind": "ensemble spread (conditional variability)",
                "variance": variance,
                "std": std,
            },
            alternative_futures=int(slice_values.shape[0]),
            limitations=[
                "ensemble spread is not a calibrated interval unless calibration is verified",
            ],
        )

    # -- Step 42 ---------------------------------------------------------------
    def sensitivity_report(
        self,
        function: Callable[[np.ndarray], np.ndarray],
        parameters: np.ndarray,
        parameter_names: Sequence[str],
    ) -> UncertaintyReport:
        """Sensitivity and identifiability analysis report."""
        result = sensitivity_analysis(function, parameters)
        flags = identifiability_flags(result["normalized_jacobian"], parameter_names)
        dominant = [
            name for name, norm in zip(parameter_names, result["column_norms"]) if norm > 0.1
        ]
        return UncertaintyReport(
            method="sensitivity_analysis",
            epistemic={
                "kind": "parameter sensitivity (local)",
                "normalized_column_norms": {
                    name: float(norm) for name, norm in zip(parameter_names, result["column_norms"])
                },
                "high_influence_parameters": dominant,
                "identifiability": flags,
            },
            limitations=[
                "local analysis around the evaluated point only",
                "correlated parameters require additional analysis",
            ],
        )
