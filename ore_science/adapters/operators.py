"""Neural operator and Fourier Neural Operator adapters (Steps 25-27).

The operator implemented here is a diagonal spectral operator: input fields are mapped
to Fourier space, each retained physical-frequency mode is scaled by a learned response
function, and the result is mapped back. Because the learned mode responses are stored
as functions of *physical frequency* (Hz per metre) rather than discrete mode indices,
the operator can be applied at resolutions it was not trained on, and multi-resolution
evaluation is honest rather than assumed. The retained mode count and the periodic
boundary convention are recorded and surfaced as explicit limitations.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = ["FourierModeOperator", "OperatorFitReport"]


@dataclass(frozen=True, slots=True)
class OperatorFitReport:
    """Training report for the spectral operator."""

    training_resolution: int
    training_samples: int
    retained_modes: int
    fit_rmse: float
    resolution_transfer_evaluations: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        """Serialize the fit report."""
        return {
            "training_resolution": self.training_resolution,
            "training_samples": self.training_samples,
            "retained_modes": self.retained_modes,
            "fit_rmse": self.fit_rmse,
            "resolution_transfer_evaluations": list(self.resolution_transfer_evaluations),
        }


class FourierModeOperator:
    """Diagonal Fourier neural operator over periodic 1-D fields (Step 26).

    The learned object is an operator over functions: a(·) -> u(·) where the response
    of each frequency is a smooth function of physical frequency, parameterized on a
    frequency grid and interpolated at evaluation time. This is a linear, diagonal
    approximation of an FNO; it is disclosed as such and never presented as a general
    learned operator.
    """

    def __init__(self, *, domain_length: float, retained_modes: int, frequency_nodes: int = 64) -> None:
        """Configure the operator's physical domain and retained spectral modes."""
        if domain_length <= 0.0:
            raise ValueError("domain length must be positive")
        if retained_modes < 1:
            raise ValueError("retained mode count must be positive")
        if frequency_nodes < 4:
            raise ValueError("frequency grid needs at least four nodes")
        self.domain_length = domain_length
        self.retained_modes = retained_modes
        self.frequency_nodes = frequency_nodes
        self.frequency_grid = np.linspace(0.0, self._nyquist(), frequency_nodes)
        self.response = np.zeros(frequency_nodes, dtype=float)
        self.fit_report: OperatorFitReport | None = None
        self.model_version = "operator.fourier_diag/1.0.0"
        self.boundary_convention = "periodic"
        self.trained = False

    def _nyquist(self) -> float:
        """Highest physical frequency representable by the retained modes."""
        return self.retained_modes / self.domain_length

    def _modes_for_resolution(self, resolution: int) -> np.ndarray:
        """Return integer mode numbers for a resolution, truncated to retained modes."""
        max_mode = min(self.retained_modes, resolution // 2)
        return np.arange(0, max_mode + 1)

    def _frequencies(self, resolution: int) -> np.ndarray:
        """Convert mode numbers to physical frequencies for a resolution."""
        return self._modes_for_resolution(resolution) / self.domain_length

    def fit(
        self,
        inputs: Sequence[np.ndarray],
        targets: Sequence[np.ndarray],
    ) -> OperatorFitReport:
        """Learn per-frequency responses from (input field, target field) pairs.

        All pairs must share one resolution and length; the learned response at each
        observed frequency is the median ratio of output to input Fourier coefficients
        (real part, with a guard near zero denominators). Frequencies above the
        retained-mode cutoff are ignored, and this truncation is recorded.
        """
        if len(inputs) != len(targets) or not inputs:
            raise ValueError("fit requires equal, non-empty input/target sequences")
        resolution = inputs[0].shape[0]
        for field in (*inputs, *targets):
            if field.ndim != 1 or field.shape[0] != resolution:
                raise ValueError("training pairs must be 1-D fields of equal resolution")

        input_spectrum = np.fft.rfft(np.asarray(inputs, dtype=float), axis=1)
        target_spectrum = np.fft.rfft(np.asarray(targets, dtype=float), axis=1)
        mode_frequencies = self._frequencies(resolution)
        observed_responses: list[tuple[float, float]] = []
        for mode_index, frequency in enumerate(mode_frequencies):
            denominators = input_spectrum[:, mode_index]
            numerators = target_spectrum[:, mode_index]
            nonzero = np.abs(denominators) > 1.0e-12
            if not np.any(nonzero):
                continue
            ratios = numerators[nonzero] / denominators[nonzero]
            response = float(np.median(np.real(ratios)))
            observed_responses.append((float(frequency), response))

        if not observed_responses:
            raise ValueError("no spectral information available to fit the operator")

        observed_frequencies = np.array([item[0] for item in observed_responses])
        observed_values = np.array([item[1] for item in observed_responses])
        self.response = np.interp(self.frequency_grid, observed_frequencies, observed_values)
        self.trained = True

        reconstructed = [self.apply(field) for field in inputs]
        residuals = [
            float(np.sqrt(np.mean((prediction - target) ** 2)))
            for prediction, target in zip(reconstructed, targets)
        ]
        self.fit_report = OperatorFitReport(
            training_resolution=resolution,
            training_samples=len(inputs),
            retained_modes=self.retained_modes,
            fit_rmse=float(np.mean(residuals)),
            resolution_transfer_evaluations=(),
        )
        self.trained = True
        return self.fit_report

    def apply(self, field: np.ndarray, *, resolution: int | None = None) -> np.ndarray:
        """Apply the operator to one periodic field, optionally at a new resolution.

        ``resolution`` resamples the input field onto a denser or sparser grid before
        applying the operator, which is exactly the resolution-transfer evaluation the
        acceptance tests require. Mode amplitudes above the retained-mode cutoff are
        discarded, never silently extrapolated.
        """
        if not self.trained:
            raise ValueError("operator must be fit before apply")
        if field.ndim != 1 or field.shape[0] < 2:
            raise ValueError("fields must be one-dimensional with at least two samples")
        work = field
        if resolution is not None and resolution != field.shape[0]:
            positions = np.linspace(0.0, 1.0, field.shape[0], endpoint=False)
            target_positions = np.linspace(0.0, 1.0, resolution, endpoint=False)
            work = np.interp(target_positions, positions, field, period=1.0)
        spectrum = np.fft.rfft(work)
        mode_frequencies = self._frequencies(work.shape[0])
        responses = np.interp(mode_frequencies, self.frequency_grid, self.response)
        spectrum = spectrum[: len(mode_frequencies)] * responses
        full = np.zeros(work.shape[0] // 2 + 1, dtype=complex)
        full[: len(spectrum)] = spectrum
        return np.fft.irfft(full, n=work.shape[0])

    def evaluate_resolution_transfer(
        self, reference_pairs: Sequence[tuple[np.ndarray, np.ndarray]],
        resolutions: Sequence[int],
    ) -> dict[str, Any]:
        """Evaluate the operator at multiple discretizations against reference pairs.

        Each pair is (input, reference output) at the pair's own resolution; the
        operator is applied at each requested resolution and the RMSE against the
        resampled reference is reported per resolution.
        """
        if not resolutions:
            raise ValueError("at least one evaluation resolution is required")
        results: dict[str, Any] = {}
        for resolution in resolutions:
            errors = []
            for field, reference in reference_pairs:
                prediction = self.apply(field, resolution=resolution)
                if reference.shape[0] != resolution:
                    positions = np.linspace(0.0, 1.0, reference.shape[0], endpoint=False)
                    target_positions = np.linspace(0.0, 1.0, resolution, endpoint=False)
                    reference = np.interp(target_positions, positions, reference, period=1.0)
                errors.append(float(np.sqrt(np.mean((prediction - reference) ** 2))))
            results[str(resolution)] = {
                "rmse": float(np.mean(errors)),
                "max_error": float(np.max(errors)),
            }
        if self.fit_report is not None:
            self.fit_report = OperatorFitReport(
                training_resolution=self.fit_report.training_resolution,
                training_samples=self.fit_report.training_samples,
                retained_modes=self.fit_report.retained_modes,
                fit_rmse=self.fit_report.fit_rmse,
                resolution_transfer_evaluations=tuple(resolutions),
            )
        return results

    def to_dict(self) -> dict[str, Any]:
        """Serialize the operator identity, retained modes, and boundary convention."""
        return {
            "model_version": self.model_version,
            "family": "neural_operator",
            "domain_length": self.domain_length,
            "retained_modes": self.retained_modes,
            "boundary_convention": self.boundary_convention,
            "trained": self.trained,
            "fit_report": self.fit_report.to_dict() if self.fit_report else None,
            "disclosure": (
                "Linear diagonal spectral operator, not a general learned FNO; "
                "resolution transfer is evaluated, never assumed."
            ),
        }
