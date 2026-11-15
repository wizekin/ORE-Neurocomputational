"""Reusable signal, orthographic, probabilistic, and mathematical operations."""

from __future__ import annotations

from collections import Counter
from typing import Iterable, Sequence

import numpy as np
from scipy.ndimage import gaussian_filter, convolve


def stable_softmax(values: np.ndarray, axis: int = -1) -> np.ndarray:
    """Compute numerically stable softmax probabilities along one axis."""
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("softmax input must be non-empty and finite")
    shifted = array - np.max(array, axis=axis, keepdims=True)
    exponentials = np.exp(np.clip(shifted, -80.0, 80.0))
    return (exponentials / np.maximum(exponentials.sum(axis=axis, keepdims=True), 1e-12)).astype(np.float32)


def normalize_probabilities(values: np.ndarray, axis: int = -1, epsilon: float = 1e-12) -> np.ndarray:
    """Clamp negative numerical noise and normalize non-zero probability rows."""
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("probability values must be non-empty and finite")
    array = np.maximum(array, 0.0)
    totals = np.sum(array, axis=axis, keepdims=True)
    if np.any(totals <= epsilon):
        raise ValueError("each probability distribution must have positive mass")
    return (array / np.maximum(totals, epsilon)).astype(np.float32)


def probability_entropy(probabilities: np.ndarray, axis: int = -1, epsilon: float = 1e-12) -> np.ndarray:
    """Calculate Shannon entropy H(p) = -sum(p log p) for each distribution."""
    p = normalize_probabilities(probabilities, axis=axis, epsilon=epsilon).astype(np.float64)
    return (-np.sum(p * np.log(np.maximum(p, epsilon)), axis=axis)).astype(np.float32)


def bayesian_posterior(likelihood: np.ndarray, prior: np.ndarray, epsilon: float = 1e-12) -> np.ndarray:
    """Combine a likelihood matrix with a class prior using Bayes' rule."""
    likelihoods = normalize_probabilities(likelihood, axis=-1, epsilon=epsilon).astype(np.float64)
    prior_values = normalize_probabilities(np.asarray(prior, dtype=np.float64), axis=-1, epsilon=epsilon)
    if prior_values.ndim != 1 or likelihoods.shape[-1] != prior_values.size:
        raise ValueError("prior must be one-dimensional and match the likelihood class dimension")
    joint = likelihoods * prior_values[None, :]
    return normalize_probabilities(joint, axis=-1, epsilon=epsilon)


def precision_weighted_prediction_error(
    observed: np.ndarray,
    predicted: np.ndarray,
    variance: np.ndarray | float,
    epsilon: float = 1e-8,
) -> dict[str, np.ndarray | float]:
    """Compute residuals, inverse-variance precision, and weighted quadratic error."""
    target = np.asarray(observed, dtype=np.float64)
    estimate = np.asarray(predicted, dtype=np.float64)
    uncertainty = np.asarray(variance, dtype=np.float64)
    if target.shape != estimate.shape or not np.isfinite(target).all() or not np.isfinite(estimate).all():
        raise ValueError("observed and predicted arrays must be finite and have identical shapes")
    if not np.isfinite(uncertainty).all() or np.any(uncertainty < 0):
        raise ValueError("variance must be finite and non-negative")
    precision = 1.0 / np.maximum(uncertainty, epsilon)
    error = target - estimate
    weighted = precision * error
    energy = float(0.5 * np.sum(precision * np.square(error)))
    return {
        "error": error.astype(np.float32),
        "precision": np.asarray(precision, dtype=np.float32),
        "weighted_error": weighted.astype(np.float32),
        "weighted_error_energy": energy,
    }


def word_surprisal(probability: float | np.ndarray, epsilon: float = 1e-12) -> np.ndarray:
    """Return Shannon surprisal in nats, -log(p), for a supplied probability."""
    values = np.asarray(probability, dtype=np.float64)
    if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
        raise ValueError("probability must lie in [0, 1]")
    return (-np.log(np.maximum(values, epsilon))).astype(np.float32)


def information_gain(prior: np.ndarray, posterior: np.ndarray, epsilon: float = 1e-12) -> np.ndarray:
    """Measure entropy reduction from prior to posterior for each row."""
    prior_p = normalize_probabilities(prior, axis=-1, epsilon=epsilon)
    posterior_p = normalize_probabilities(posterior, axis=-1, epsilon=epsilon)
    if prior_p.shape != posterior_p.shape:
        raise ValueError("prior and posterior must have identical shapes")
    return (probability_entropy(prior_p, axis=-1) - probability_entropy(posterior_p, axis=-1)).astype(np.float32)


def difference_of_gaussians(
    image: np.ndarray, sigma_center: float = 1.0, sigma_surround: float = 2.0
) -> np.ndarray:
    """Compute a center-minus-surround spatial response with two Gaussian scales."""
    frame = np.asarray(image, dtype=np.float64)
    if frame.ndim != 2 or not np.isfinite(frame).all():
        raise ValueError("image must be a finite two-dimensional array")
    if sigma_center <= 0 or sigma_surround <= sigma_center:
        raise ValueError("surround sigma must exceed a positive center sigma")
    center = gaussian_filter(frame, sigma_center, mode="reflect")
    surround = gaussian_filter(frame, sigma_surround, mode="reflect")
    return (center - surround).astype(np.float32)


def gabor_kernel(
    size: int, orientation_radians: float, spatial_frequency: float,
    sigma: float, aspect_ratio: float = 0.5, phase_radians: float = 0.0,
) -> np.ndarray:
    """Create a zero-mean, unit-norm Gabor receptive-field approximation."""
    if size < 3 or sigma <= 0 or spatial_frequency <= 0 or aspect_ratio <= 0:
        raise ValueError("Gabor size and scale parameters must be positive")
    axis = np.arange(size, dtype=np.float64) - (size - 1.0) / 2.0
    xx, yy = np.meshgrid(axis, axis)
    cosine = np.cos(orientation_radians)
    sine = np.sin(orientation_radians)
    x_rot = xx * cosine + yy * sine
    y_rot = -xx * sine + yy * cosine
    envelope = np.exp(-0.5 * (np.square(x_rot / sigma) + np.square(y_rot / (sigma / aspect_ratio))))
    carrier = np.cos(2.0 * np.pi * spatial_frequency * x_rot + phase_radians)
    kernel = envelope * carrier
    kernel -= np.mean(kernel)
    norm = float(np.linalg.norm(kernel))
    if norm <= 1e-12:
        raise ValueError("Gabor kernel has near-zero energy")
    return (kernel / norm).astype(np.float32)


def orientation_energy_bank(
    image: np.ndarray, orientations: Sequence[float] = (0.0, 0.785398, 1.570796, 2.356194),
    spatial_frequency: float = 0.12, sigma: float = 2.0, kernel_size: int = 15,
) -> np.ndarray:
    """Return quadrature Gabor energy maps [orientation, row, column]."""
    frame = np.asarray(image, dtype=np.float64)
    if frame.ndim != 2 or not np.isfinite(frame).all():
        raise ValueError("image must be a finite two-dimensional array")
    if len(orientations) == 0:
        raise ValueError("at least one orientation is required")
    responses = []
    for orientation in orientations:
        even = gabor_kernel(kernel_size, float(orientation), spatial_frequency, sigma, phase_radians=0.0)
        odd = gabor_kernel(kernel_size, float(orientation), spatial_frequency, sigma, phase_radians=np.pi / 2.0)
        even_response = convolve(frame, even, mode="reflect")
        odd_response = convolve(frame, odd, mode="reflect")
        responses.append(np.sqrt(np.square(even_response) + np.square(odd_response)))
    return np.asarray(responses, dtype=np.float32)


def divisive_population_normalization(
    activity: np.ndarray, pool_sigma: float = 1.0, epsilon: float = 1e-6, exponent: float = 2.0
) -> np.ndarray:
    """Normalize channel activity by a locally pooled population-energy denominator."""
    values = np.asarray(activity, dtype=np.float64)
    if values.ndim < 2 or not np.isfinite(values).all():
        raise ValueError("activity must be a finite array with a channel axis")
    if pool_sigma < 0 or epsilon <= 0 or exponent <= 0:
        raise ValueError("normalization parameters are outside their valid ranges")
    pooled = np.sum(np.power(np.maximum(values, 0.0), exponent), axis=0)
    denominator = np.power(
        epsilon + gaussian_filter(pooled, pool_sigma, mode="reflect"), 1.0 / exponent
    )
    return (np.maximum(values, 0.0) / np.maximum(denominator[None, ...], epsilon)).astype(np.float32)


def spatial_pool(feature_maps: np.ndarray, grid_shape: tuple[int, int] = (4, 4)) -> np.ndarray:
    """Mean-pool channel-first feature maps into a fixed spatial grid."""
    maps = np.asarray(feature_maps, dtype=np.float64)
    if maps.ndim != 3 or not np.isfinite(maps).all():
        raise ValueError("feature_maps must have shape [channel, height, width] and be finite")
    grid_y, grid_x = grid_shape
    if grid_y < 1 or grid_x < 1 or grid_y > maps.shape[1] or grid_x > maps.shape[2]:
        raise ValueError("grid_shape must fit inside feature map dimensions")
    row_edges = np.linspace(0, maps.shape[1], grid_y + 1, dtype=int)
    col_edges = np.linspace(0, maps.shape[2], grid_x + 1, dtype=int)
    output = np.zeros((maps.shape[0], grid_y, grid_x), dtype=np.float32)
    for row in range(grid_y):
        for col in range(grid_x):
            patch = maps[:, row_edges[row]:row_edges[row + 1], col_edges[col]:col_edges[col + 1]]
            output[:, row, col] = np.mean(patch, axis=(1, 2))
    return output


def mahalanobis_distances(
    samples: np.ndarray, mean: np.ndarray, covariance: np.ndarray, ridge: float = 1e-6
) -> np.ndarray:
    """Measure squared Mahalanobis distance from each sample to a prototype."""
    x = np.atleast_2d(np.asarray(samples, dtype=np.float64))
    center = np.asarray(mean, dtype=np.float64)
    cov = np.asarray(covariance, dtype=np.float64)
    if center.ndim != 1 or x.shape[1] != center.size or cov.shape != (center.size, center.size):
        raise ValueError("samples, mean, and covariance dimensions are incompatible")
    if not np.isfinite(x).all() or not np.isfinite(center).all() or not np.isfinite(cov).all():
        raise ValueError("Mahalanobis inputs must be finite")
    regularized = cov + ridge * np.eye(center.size)
    inverse = np.linalg.pinv(regularized)
    delta = x - center[None, :]
    return np.maximum(np.einsum("bi,ij,bj->b", delta, inverse, delta), 0.0).astype(np.float32)


def cosine_similarity(left: np.ndarray, right: np.ndarray, epsilon: float = 1e-12) -> float:
    """Calculate cosine similarity while guarding against zero-norm vectors."""
    a = np.asarray(left, dtype=np.float64).ravel()
    b = np.asarray(right, dtype=np.float64).ravel()
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("cosine vectors must be finite and have equal dimensions")
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / max(denominator, epsilon))


def ordinal_position_code(length: int) -> np.ndarray:
    """Create an explicit one-hot ordinal code for each sequence position."""
    if length < 1:
        raise ValueError("length must be positive")
    return np.eye(length, dtype=np.float32)


def ordered_ngrams(sequence: Iterable[str], n: int = 2) -> dict[tuple[str, ...], int]:
    """Count ordered contiguous n-grams without discarding symbol position."""
    symbols = tuple(sequence)
    if n < 1:
        raise ValueError("n must be positive")
    return dict(Counter(tuple(symbols[index:index + n]) for index in range(max(0, len(symbols) - n + 1))))


def edit_distance(left: Sequence[str], right: Sequence[str]) -> int:
    """Compute Levenshtein insertion, deletion, and substitution distance."""
    a, b = tuple(left), tuple(right)
    previous = list(range(len(b) + 1))
    for i, item_a in enumerate(a, start=1):
        current = [i]
        for j, item_b in enumerate(b, start=1):
            current.append(min(
                current[-1] + 1,
                previous[j] + 1,
                previous[j - 1] + (item_a != item_b),
            ))
        previous = current
    return previous[-1]


def normalized_edit_similarity(left: Sequence[str], right: Sequence[str]) -> float:
    """Convert edit distance to a bounded similarity score in [0, 1]."""
    denominator = max(len(left), len(right), 1)
    return float(1.0 - edit_distance(left, right) / denominator)


__all__ = [
    "stable_softmax", "normalize_probabilities", "probability_entropy",
    "bayesian_posterior", "precision_weighted_prediction_error",
    "word_surprisal", "information_gain", "difference_of_gaussians",
    "gabor_kernel", "orientation_energy_bank", "divisive_population_normalization",
    "spatial_pool", "mahalanobis_distances", "cosine_similarity",
    "ordinal_position_code", "ordered_ngrams", "edit_distance",
    "normalized_edit_similarity",
]
