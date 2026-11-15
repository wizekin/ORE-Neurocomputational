"""Operation 1: Gabor-inspired early visual feature extraction."""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy import ndimage as ndi

from .contracts import BNBEAConfig


def to_grayscale(image: Any) -> np.ndarray:
    """Convert a grayscale, RGB or RGBA image array into normalized float grayscale values."""
    array = np.asarray(image, dtype=np.float64)
    if array.ndim == 3:
        rgb = array[..., :3]
        array = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    if array.ndim != 2 or array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("image must be a finite, non-empty grayscale or RGB array")
    minimum, maximum = float(np.min(array)), float(np.max(array))
    if maximum > minimum:
        array = (array - minimum) / (maximum - minimum)
    else:
        array = np.zeros_like(array)
    return array


def normalize_contrast(image: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    """Apply global z-score contrast normalization as an engineering approximation."""
    values = np.asarray(image, dtype=np.float64)
    return (values - float(np.mean(values))) / (float(np.std(values)) + epsilon)


def gabor_kernel(
    size: int, theta: float, frequency: float, sigma: float, gamma: float, phase: float
) -> np.ndarray:
    """Construct a zero-mean, unit-energy 2-D Gabor kernel for one receptive-field preference."""
    radius = size // 2
    yy, xx = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    x_rot = xx * np.cos(theta) + yy * np.sin(theta)
    y_rot = -xx * np.sin(theta) + yy * np.cos(theta)
    envelope = np.exp(-(x_rot**2 + (gamma**2) * y_rot**2) / (2.0 * sigma**2))
    kernel = envelope * np.cos(2.0 * np.pi * frequency * x_rot + phase)
    kernel -= np.mean(kernel)
    norm = float(np.linalg.norm(kernel))
    return kernel / max(norm, 1e-12)


class GaborVisualProcessor:
    """Extract V1-inspired orientation and spatial-frequency feature maps from a text image."""

    def __init__(self, config: BNBEAConfig | None = None) -> None:
        """Store the numerical filter-bank configuration used for each visual sample."""
        self.config = config or BNBEAConfig()

    def process(self, image: Any) -> dict[str, Any]:
        """Normalize an image and calculate quadrature Gabor energy plus local feature summaries."""
        grayscale = to_grayscale(image)
        normalized = normalize_contrast(grayscale)
        maps: dict[str, np.ndarray] = {}
        summaries: list[dict[str, float]] = []
        for orientation_index, theta in enumerate(self.config.orientations):
            for frequency_index, frequency in enumerate(self.config.spatial_frequencies):
                even = gabor_kernel(
                    self.config.gabor_size, theta, frequency,
                    self.config.gabor_sigma, self.config.gabor_gamma, 0.0,
                )
                odd = gabor_kernel(
                    self.config.gabor_size, theta, frequency,
                    self.config.gabor_sigma, self.config.gabor_gamma, np.pi / 2.0,
                )
                response_even = ndi.convolve(normalized, even, mode="reflect")
                response_odd = ndi.convolve(normalized, odd, mode="reflect")
                energy = np.sqrt(response_even**2 + response_odd**2)
                key = f"theta_{orientation_index}_frequency_{frequency_index}"
                maps[key] = energy
                summaries.append({
                    "orientation_rad": float(theta),
                    "spatial_frequency": float(frequency),
                    "mean_energy": float(np.mean(energy)),
                    "max_energy": float(np.max(energy)),
                })
        stack = np.stack(list(maps.values()), axis=0)
        rectified = np.maximum(stack - np.quantile(stack, 0.25), 0.0)
        denominator = 1e-6 + ndi.gaussian_filter(np.sum(rectified, axis=0), sigma=2.0)
        normalized_stack = rectified / denominator[None, ...]
        feature_maps = {
            key: normalized_stack[index]
            for index, key in enumerate(maps)
        }
        return {
            "normalized_image": normalized,
            "raw_orientation_energy_maps": maps,
            "feature_maps": feature_maps,
            "feature_stack": normalized_stack,
            "summary": summaries,
            "mathematics": {
                "normalization": "(I - mean(I)) / (std(I) + epsilon)",
                "gabor": "exp(-(x_rot^2 + gamma^2*y_rot^2)/(2*sigma^2))*cos(2*pi*f*x_rot+phase)",
                "energy": "sqrt(response_even^2 + response_odd^2)",
                "normalization_note": "divisive local gain control; not literal retinal computation",
            },
            "biological_analogue": (
                "Retinal/LGN contrast-sensitive processing and V1-like orientation/spatial-frequency "
                "selectivity; mathematical filters are a functional approximation."
            ),
        }


def extract_region_feature_vector(feature_stack: np.ndarray, region_mask: np.ndarray) -> np.ndarray:
    """Summarize local orientation energies and region geometry into a fixed-length feature vector."""
    channels, height, width = feature_stack.shape
    rows, columns = np.nonzero(region_mask)
    if rows.size == 0:
        raise ValueError("region_mask must contain at least one pixel")
    values = feature_stack[:, region_mask]
    orientation_means = np.mean(values, axis=1)
    orientation_stds = np.std(values, axis=1)
    y_center = float(np.mean(rows) / max(height - 1, 1))
    x_center = float(np.mean(columns) / max(width - 1, 1))
    width_ratio = float((np.max(columns) - np.min(columns) + 1) / max(width, 1))
    height_ratio = float((np.max(rows) - np.min(rows) + 1) / max(height, 1))
    density = float(rows.size / max(height_ratio * width_ratio * height * width, 1.0))
    horizontal_projection = np.sum(region_mask, axis=1)
    vertical_projection = np.sum(region_mask, axis=0)
    feature_tail = np.array([
        x_center, y_center, width_ratio, height_ratio, density,
        float(np.std(horizontal_projection)), float(np.std(vertical_projection)),
    ])
    return np.concatenate((orientation_means, orientation_stds, feature_tail)).astype(np.float64)
