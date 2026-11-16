"""Numerical primitives shared by the six sensory-perception stages."""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from scipy import signal


def sigmoid(values: np.ndarray | float) -> np.ndarray:
    """Apply a numerically stable logistic nonlinearity as a bounded population activation proxy."""
    x = np.asarray(values, dtype=np.float64)
    x = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x))


def softmax(values: np.ndarray, axis: int = -1, temperature: float = 1.0) -> np.ndarray:
    """Normalize competing neural-candidate scores into a stable probability-like distribution."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    x = np.asarray(values, dtype=np.float64) / temperature
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(np.clip(x, -60.0, 0.0))
    return exp_x / np.maximum(np.sum(exp_x, axis=axis, keepdims=True), 1e-12)


def frame_signal(values: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    """Slice a one-dimensional waveform into overlapping frames without discarding the final partial frame."""
    x = np.asarray(values, dtype=np.float64).ravel()
    if frame_length < 2 or hop_length < 1:
        raise ValueError("frame_length must be >= 2 and hop_length >= 1")
    if x.size < frame_length:
        x = np.pad(x, (0, frame_length - x.size))
    count = 1 + int(np.ceil((x.size - frame_length) / hop_length))
    required = (count - 1) * hop_length + frame_length
    if required > x.size:
        x = np.pad(x, (0, required - x.size))
    indices = np.arange(frame_length)[None, :] + hop_length * np.arange(count)[:, None]
    return x[indices]


def normalized_autocorrelation(frame: np.ndarray, min_lag: int = 1, max_lag: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return non-negative-lag normalized autocorrelation for periodicity and pitch evidence."""
    x = np.asarray(frame, dtype=np.float64).ravel()
    x = x - float(np.mean(x)) if x.size else x
    if x.size < 3:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.int64)
    ac = signal.correlate(x, x, mode="full", method="fft")[x.size - 1:]
    ac = ac / max(float(ac[0]), 1e-12)
    stop = min(max_lag or (x.size - 1), x.size - 1)
    start = max(0, min_lag)
    lags = np.arange(start, stop + 1, dtype=np.int64)
    return ac[lags], lags


def otsu_threshold(values: np.ndarray) -> float:
    """Estimate a two-class intensity threshold by maximizing between-class variance."""
    x = np.asarray(values, dtype=np.float64)
    if x.size == 0:
        return 0.5
    hist, edges = np.histogram(x, bins=128, range=(float(np.min(x)), float(np.max(x))))
    if np.sum(hist) == 0 or edges.size < 2:
        return float(np.mean(x))
    probabilities = hist.astype(np.float64) / np.sum(hist)
    centers = (edges[:-1] + edges[1:]) / 2.0
    omega = np.cumsum(probabilities)
    mean = np.cumsum(probabilities * centers)
    total_mean = mean[-1]
    denominator = omega * (1.0 - omega)
    between = np.zeros_like(denominator)
    valid = denominator > 1e-12
    between[valid] = (total_mean * omega[valid] - mean[valid]) ** 2 / denominator[valid]
    return float(centers[int(np.argmax(between))])


def safe_summary(values: np.ndarray) -> dict[str, object]:
    """Summarize a numerical representation without storing its raw pixels or sample values."""
    x = np.asarray(values, dtype=np.float64)
    finite = x[np.isfinite(x)]
    return {
        "shape": list(x.shape),
        "mean": float(np.mean(finite)) if finite.size else 0.0,
        "std": float(np.std(finite)) if finite.size else 0.0,
        "min": float(np.min(finite)) if finite.size else 0.0,
        "max": float(np.max(finite)) if finite.size else 0.0,
        "finite_fraction": float(finite.size / max(x.size, 1)),
    }


def grayscale_image(image: object, epsilon: float = 1e-8) -> np.ndarray:
    """Convert grayscale/RGB(A) pixels into normalized luminance for retinal/V1-inspired processing."""
    array = np.asarray(image, dtype=np.float64)
    if array.ndim == 3:
        array = array[..., :3]
        if array.shape[2] == 1:
            array = array[..., 0]
        else:
            array = 0.2126 * array[..., 0] + 0.7152 * array[..., 1] + 0.0722 * array[..., 2]
    if array.ndim != 2 or array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("image must be a finite non-empty grayscale or RGB(A) array")
    low, high = float(np.min(array)), float(np.max(array))
    if high - low <= epsilon:
        return np.zeros_like(array)
    if low < 0.0 or high > 1.0:
        array = (array - low) / (high - low)
    return np.clip(array, 0.0, 1.0)


def gabor_kernel(size: int, theta: float, frequency: float, sigma: float = 2.8, gamma: float = 0.7, phase: float = 0.0) -> np.ndarray:
    """Build a zero-mean normalized oriented Gabor filter as a proxy for V1 receptive-field tuning."""
    radius = size // 2
    yy, xx = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    xr = xx * np.cos(theta) + yy * np.sin(theta)
    yr = -xx * np.sin(theta) + yy * np.cos(theta)
    envelope = np.exp(-(xr**2 + gamma**2 * yr**2) / (2.0 * sigma**2))
    kernel = envelope * np.cos(2.0 * np.pi * frequency * xr + phase)
    kernel -= float(np.mean(kernel))
    return kernel / max(float(np.linalg.norm(kernel)), 1e-12)


def robust_zscore(values: np.ndarray, baseline: tuple[float, float] | None = None, epsilon: float = 1e-8) -> tuple[np.ndarray, tuple[float, float]]:
    """Normalize speaker-sensitive measurements using supplied or sample-derived center and scale."""
    x = np.asarray(values, dtype=np.float64)
    finite = x[np.isfinite(x)]
    if baseline is None:
        center = float(np.median(finite)) if finite.size else 0.0
        scale = float(1.4826 * np.median(np.abs(finite - center))) if finite.size else 1.0
        if scale <= epsilon:
            scale = float(np.std(finite)) if finite.size else 1.0
    else:
        center, scale = float(baseline[0]), float(baseline[1])
    scale = max(scale, epsilon)
    normalized = (x - center) / scale
    return np.clip(normalized, -8.0, 8.0), (center, scale)
