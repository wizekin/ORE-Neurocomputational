"""Independent validation engine: dimensions, convergence, invariants, baselines (Steps 35-38, 44).

Validation is separate from the models it checks. Dimensional validation runs before
expensive execution; numerical convergence varies resolution and reports converged or
not-converged explicitly; conservation checks detect perturbed solutions; baseline
comparison records discrepancy against analytic or numerical references; and the
domain review gate selects the required checks by task class. A prediction is not
proof: validation reports are attached to results rather than absorbed into them.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from .contracts import (
    Constraint,
    EquationModel,
    ResultStatus,
    ScientificDomain,
    ScientificTask,
    ValidationCheck,
    ValidationReport,
)
from .units import dimensional_homogeneity

__all__ = ["ValidationEngine", "richardson_convergence"]


_GATES_BY_DOMAIN: dict[ScientificDomain, tuple[str, ...]] = {
    ScientificDomain.PHYSICAL_SIMULATION: (
        "dimensional_consistency",
        "numerical_convergence",
        "conservation",
        "baseline_comparison",
    ),
    ScientificDomain.FORECASTING: (
        "dimensional_consistency",
        "calibration_metrics",
        "ensemble_statistics_preserved",
    ),
    ScientificDomain.LOCALIZATION: (
        "factor_graph_consistency",
        "convergence_report",
        "frame_consistency",
    ),
    ScientificDomain.MATERIALS: (
        "charge_balance",
        "composition_bounds",
        "geometry_checks",
        "duplicate_detection",
    ),
    ScientificDomain.MOLECULAR_STRUCTURE: ("structure_checks",),
    ScientificDomain.MOLECULAR_INTERACTION: ("interaction_class_compatibility", "structure_checks"),
    ScientificDomain.PROTEIN_DESIGN: ("candidate_structure_checks",),
    ScientificDomain.GEOMETRIC: ("equivariance_tests", "frame_consistency"),
}


def richardson_convergence(
    values: Sequence[float], refinement_factor: float = 2.0
) -> dict[str, Any]:
    """Estimate convergence order and the discretization error ratio.

    Given solutions at successively refined resolutions, the observed ratio
    |v_coarse - v_medium| / |v_medium - v_fine| must exceed 1 for the error to be
    shrinking under refinement (it approaches ``refinement_factor**p`` for a
    order-p method). Ratios at or below 1 indicate stagnation or divergence. The
    outcome is stated explicitly as ``converged`` or ``not_converged``; the last
    iterate is never silently returned as a converged result.
    """
    if len(values) < 2:
        return {
            "converged": False,
            "status": "insufficient_data",
            "detail": "at least two refinement levels are required",
        }
    errors = [
        abs(values[index] - values[index + 1]) for index in range(len(values) - 1)
    ]
    if all(error == 0.0 for error in errors):
        return {
            "converged": True,
            "status": "converged",
            "detail": "solutions identical across refinements",
            "error_ratios": [],
        }
    ratios = [
        (errors[index] / errors[index + 1]) if errors[index + 1] > 0.0 else float("inf")
        for index in range(len(errors) - 1)
    ]
    last_ratio = ratios[-1] if ratios else 0.0
    converged = last_ratio > 1.0
    return {
        "converged": bool(converged),
        "status": "converged" if converged else "not_converged",
        "successive_errors": errors,
        "error_ratios": ratios,
        "last_ratio": last_ratio,
        "threshold": 1.0,
        "detail": (
            f"last error ratio {last_ratio:.3g}; errors must shrink (ratio > 1) under "
            f"refinement, approaching {refinement_factor}**p for an order-p method"
        ),
    }


class ValidationEngine:
    """Aggregates dimensional, physical, numerical, and baseline validation (Section 3.F)."""

    # -- Step 35 ---------------------------------------------------------------
    def dimensional_validation(self, equations: Sequence[EquationModel]) -> ValidationReport:
        """Check unit consistency and term structure of every governing equation.

        Domains that declare no governing equation (forecasting, localization,
        materials, geometric) pass with an explicit statement that only variable-unit
        validation applies; equations that are declared are always checked.
        """
        report = ValidationReport()
        if not equations:
            report.add(
                ValidationCheck(
                    name="dimensional_consistency",
                    passed=True,
                    detail=(
                        "no governing equation declared for this domain class; "
                        "variable units validated individually"
                    ),
                )
            )
            return report
        for equation in equations:
            homogeneous, message = dimensional_homogeneity(equation.term_quantities())
            report.add(
                ValidationCheck(
                    name=f"dimensional_consistency:{equation.equation_id}",
                    passed=homogeneous,
                    detail=message or f"{equation.form} is dimensionally homogeneous",
                )
            )
        return report

    def variable_domain_validation(self, task: ScientificTask) -> ValidationReport:
        """Check that observed variables lie in physically admissible ranges."""
        report = ValidationReport()
        for variable in task.variables:
            if variable.value is None:
                continue
            unit_symbol = variable.unit.symbol
            value = variable.value
            problems = []
            if unit_symbol in ("K", "°C") and value < 0.0:
                problems.append("absolute temperature cannot be negative")
            if unit_symbol in ("m", "m2", "m3", "s", "kg") and value < 0.0:
                problems.append(f"{unit_symbol} quantities cannot be negative in this model")
            if variable.symbol == "alpha" and value == 0.0:
                problems.append("diffusivity must be positive")
            report.add(
                ValidationCheck(
                    name=f"variable_domain:{variable.symbol}",
                    passed=not problems,
                    detail="; ".join(problems) or f"{variable.symbol} = {value:g} {unit_symbol} admissible",
                )
            )
        return report

    # -- Step 36 ---------------------------------------------------------------
    def numerical_convergence(
        self, probe_values: Sequence[float], quantity_name: str
    ) -> ValidationReport:
        """Report whether a quantity is stable under resolution refinement."""
        report = ValidationReport()
        outcome = richardson_convergence(probe_values)
        report.add(
            ValidationCheck(
                name=f"numerical_convergence:{quantity_name}",
                passed=bool(outcome["converged"]),
                detail=(
                    f"{quantity_name}: {outcome['status']} - {outcome.get('detail', '')}"
                ),
            )
        )
        return report

    # -- Step 37 ---------------------------------------------------------------
    def conservation_checks(
        self,
        fields: Sequence[np.ndarray],
        x_grid: np.ndarray,
        *,
        tolerance: float,
        constraint: Constraint | None = None,
    ) -> ValidationReport:
        """Check that integrated field energy decays monotonically (no sources).

        For the heat equation without sources, the integral of u over the domain must
        be non-increasing in time. A deliberately perturbed solution violates the
        monotonicity or the decay tolerance and fails the check.
        """
        report = ValidationReport()
        integrals = [float(np.trapezoid(field, x_grid)) for field in fields]
        violations = []
        for index in range(1, len(integrals)):
            change = integrals[index] - integrals[index - 1]
            if change > tolerance:
                violations.append(
                    f"integrated quantity increased by {change:.3e} between steps "
                    f"{index - 1} and {index} (tolerance {tolerance:.1e})"
                )
        decayed = integrals[-1] <= integrals[0] + tolerance
        name = (
            f"conservation:{constraint.kind}" if constraint else "conservation:monotonic_decay"
        )
        report.add(
            ValidationCheck(
                name=name,
                passed=(not violations) and decayed,
                detail="; ".join(violations)
                or (
                    f"integrated quantity decays from {integrals[0]:.4g} to "
                    f"{integrals[-1]:.4g} within tolerance {tolerance:.1e}"
                ),
            )
        )
        return report

    # -- Step 38 ---------------------------------------------------------------
    def baseline_comparison(
        self,
        prediction: np.ndarray,
        baseline: np.ndarray,
        *,
        quantity_name: str,
        tolerance: float,
        baseline_identity: str,
    ) -> ValidationReport:
        """Record the discrepancy between a prediction and an independent baseline."""
        report = ValidationReport()
        if prediction.shape != baseline.shape:
            report.add(
                ValidationCheck(
                    name=f"baseline_comparison:{quantity_name}",
                    passed=False,
                    detail=(
                        f"shape mismatch: prediction {prediction.shape} vs baseline "
                        f"{baseline.shape}; resampling required before comparison"
                    ),
                )
            )
            return report
        rmse = float(np.sqrt(np.mean((prediction - baseline) ** 2)))
        scale = max(float(np.max(np.abs(baseline))), 1.0e-12)
        relative = rmse / scale
        report.add(
            ValidationCheck(
                name=f"baseline_comparison:{quantity_name}",
                passed=relative <= tolerance,
                detail=(
                    f"RMSE {rmse:.4g}, relative {relative:.3g} vs tolerance {tolerance:.1g} "
                    f"against {baseline_identity}"
                ),
            )
        )
        return report

    # -- Step 44: domain review gate -------------------------------------------
    def domain_review_gate(
        self, domain: ScientificDomain, report: ValidationReport
    ) -> ValidationReport:
        """Verify that the gate required by the task class has passing checks."""
        required = _GATES_BY_DOMAIN.get(domain, ("dimensional_consistency",))
        present = {check.name.split(":")[0] for check in report.checks}
        gate_passed = all(any(name.startswith(requirement) for name in present) for requirement in required)
        missing = [item for item in required if not any(name.startswith(item) for name in present)]
        gate = ValidationCheck(
            name=f"review_gate:{domain.value}",
            passed=gate_passed and report.passed,
            detail=(
                ("required checks present and passing: " + ", ".join(required))
                if gate_passed and report.passed
                else (
                    "missing required checks: " + ", ".join(missing)
                    if missing
                    else "required checks present but failed: "
                    + ", ".join(
                        check.name
                        for check in report.checks
                        if check.severity == "required" and not check.passed
                    )
                )
            ),
        )
        gated = ValidationReport(checks=list(report.checks) + [gate], warnings=list(report.warnings))
        return gated

    # -- Status ladder (Step 43) ------------------------------------------------
    @staticmethod
    def status_after_validation(report: ValidationReport, current: ResultStatus) -> ResultStatus:
        """Promote to COMPUTATIONALLY_VALIDATED only when the report passes."""
        if current is ResultStatus.PREDICTED and report.passed:
            return ResultStatus.COMPUTATIONALLY_VALIDATED
        return current
