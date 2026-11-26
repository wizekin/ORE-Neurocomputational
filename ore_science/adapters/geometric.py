"""Equivariant geometric learning interfaces and transformation tests (Steps 28-29).

Two point-set modules are provided: an SE(2)-equivariant module built from pairwise
displacement vectors (outputs rotate with the frame), and a deliberately
non-equivariant reference MLP on raw coordinates. The equivariance test protocol
measures the normalized error eps_eq = ||f(g x) - g f(x)|| / (||g f(x)|| + eps)
over representative transformations, so equivariance is verified, never assumed.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "EquivarianceTestProtocol",
    "EquivariantPointModule",
    "NonEquivariantPointModule",
    "Rotation2D",
]


@dataclass(frozen=True, slots=True)
class Rotation2D:
    """A planar rotation used as a group element of SO(2)."""

    angle: float

    def matrix(self) -> np.ndarray:
        """Return the 2x2 rotation matrix."""
        cosine, sine = math.cos(self.angle), math.sin(self.angle)
        return np.array([[cosine, -sine], [sine, cosine]])

    def act(self, points: np.ndarray) -> np.ndarray:
        """Apply the rotation to a stack of 2-D points or vectors."""
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("rotation acts on (N, 2) arrays")
        return points @ self.matrix().T


class EquivariantPointModule:
    """SE(2)-equivariant point-set processor (Step 28).

    The module computes translation-invariant pairwise displacement statistics and
    emits per-point vector features that rotate with the input frame. Rotation
    equivariance is exact by construction, but it is still tested explicitly because
    the specification forbids assuming equivariance from architecture alone.
    """

    model_version = "geometric.equivariant_points/1.0.0"

    def forward(self, points: np.ndarray) -> np.ndarray:
        """Return (N, 2) vector features equivariant under rotation and translation.

        Each output is the displacement of a point from the weighted centroid of its
        neighbourhood, which transforms as a vector under SE(2).
        """
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("module expects (N, 2) points")
        displacements = points[:, None, :] - points[None, :, :]  # (N, N, 2)
        distances = np.linalg.norm(displacements, axis=2)
        # Scale from a rotation invariant: RMS distance from the arithmetic centroid.
        centroid = points.mean(axis=0)
        scale = float(np.sqrt(np.mean(np.sum((points - centroid) ** 2, axis=1))))
        weights = np.exp(-distances / max(scale, 1.0e-9))
        weights_sum = weights.sum(axis=1, keepdims=True)
        weighted_centroids = (
            np.einsum("nm,md->nd", weights, points) / np.maximum(weights_sum, 1.0e-12)
        )
        return points - weighted_centroids

    def to_dict(self) -> dict[str, Any]:
        """Serialize the module identity and group declaration."""
        return {
            "model_version": self.model_version,
            "family": "geometric",
            "group": "SE(2)",
            "output_representation": "vector (rotates with the frame)",
            "equivariance": "by construction; verified by explicit transformation tests",
        }


class NonEquivariantPointModule:
    """Deliberately non-equivariant reference module for violation detection.

    Applies a fixed deterministic linear map to raw coordinates. Because the map is
    not rotation-covariant, f(g x) != g f(x) for generic rotations, which the
    transformation tests must detect.
    """

    model_version = "geometric.naive_mlp/1.0.0"

    def __init__(self, seed: int = 7) -> None:
        """Initialize a fixed deterministic weight matrix."""
        generator = np.random.default_rng(seed)
        self.weights = generator.normal(loc=0.0, scale=0.8, size=(2, 2))

    def forward(self, points: np.ndarray) -> np.ndarray:
        """Apply the fixed linear map to raw coordinates."""
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("module expects (N, 2) points")
        return points @ self.weights

    def to_dict(self) -> dict[str, Any]:
        """Serialize the module identity."""
        return {
            "model_version": self.model_version,
            "family": "geometric",
            "group": "none",
            "output_representation": "raw coordinates",
            "equivariance": "not designed for any group; expected to fail tests",
        }


@dataclass(frozen=True, slots=True)
class EquivarianceResult:
    """Measured normalized equivariance error for one transformation."""

    angle: float
    eps_eq: float

    def to_dict(self) -> dict[str, Any]:
        """Serialize the per-transformation result."""
        return {"angle_deg": round(math.degrees(self.angle), 2), "eps_eq": self.eps_eq}


class EquivarianceTestProtocol:
    """Normalized equivariance and invariance measurement (Step 29)."""

    def __init__(self, angles: Sequence[float] = (0.0, 30.0, 90.0, 180.0, 250.0)) -> None:
        """Configure the representative transformations."""
        if not angles:
            raise ValueError("at least one test angle is required")
        self.angles = tuple(float(angle) for angle in angles)

    def equivariance_error(
        self, module: Any, points: np.ndarray, *, eps: float = 1.0e-12
    ) -> tuple[float, list[EquivarianceResult]]:
        """Measure eps_eq = ||f(g x) - g f(x)|| / (||g f(x)|| + eps) per angle."""
        details: list[EquivarianceResult] = []
        for angle in self.angles:
            rotation = Rotation2D(math.radians(angle))
            transformed_output = module.forward(rotation.act(points))
            expected_output = rotation.act(module.forward(points))
            numerator = float(np.linalg.norm(transformed_output - expected_output))
            denominator = float(np.linalg.norm(expected_output)) + eps
            details.append(EquivarianceResult(angle=math.radians(angle), eps_eq=numerator / denominator))
        worst = max(item.eps_eq for item in details)
        return worst, details

    def invariance_error(
        self, module: Any, points: np.ndarray, *, eps: float = 1.0e-12
    ) -> tuple[float, list[EquivarianceResult]]:
        """Measure the invariance error ||f(g x) - f(x)|| / (||f(x)|| + eps) per angle.

        Used for scalar-valued or invariant outputs; a rotation-invariant energy
        should produce errors at numerical-noise level.
        """
        details: list[EquivarianceResult] = []
        base = module.forward(points)
        for angle in self.angles:
            rotation = Rotation2D(math.radians(angle))
            transformed_output = module.forward(rotation.act(points))
            numerator = float(np.linalg.norm(transformed_output - base))
            denominator = float(np.linalg.norm(base)) + eps
            details.append(
                EquivarianceResult(angle=math.radians(angle), eps_eq=numerator / denominator)
            )
        worst = max(item.eps_eq for item in details)
        return worst, details

    def to_dict(self) -> dict[str, Any]:
        """Serialize the protocol configuration."""
        return {
            "angles_deg": [round(math.degrees(angle), 2) for angle in self.angles],
            "metric": "eps_eq = ||f(g x) - g f(x)|| / (||g f(x)|| + eps)",
        }
