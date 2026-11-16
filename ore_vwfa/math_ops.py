"""Numerical primitives for visual filtering, population codes, orthographic matching, and morphology."""
from __future__ import annotations

import math
import unicodedata
from typing import Iterable, Mapping, Sequence

import numpy as np
from scipy import ndimage as ndi


def as_gray(image: np.ndarray) -> np.ndarray:
    """Convert a finite normalized grayscale/RGB/RGBA image into one luminance plane."""
    array = np.asarray(image, dtype=np.float64)
    if array.ndim == 2:
        gray = array
    elif array.ndim == 3 and array.shape[2] in (1, 3, 4):
        rgb = array[..., :3]
        if rgb.shape[2] == 1:
            gray = rgb[..., 0]
        else:
            gray = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    else:
        raise ValueError("image must be grayscale or RGB/RGBA")
    if gray.size == 0 or not np.all(np.isfinite(gray)):
        raise ValueError("image must be non-empty and finite")
    if np.min(gray) < 0.0 or np.max(gray) > 1.0:
        raise ValueError("image luminance must lie in [0, 1]")
    return gray


def softmax(values: np.ndarray, temperature: float = 1.0, axis: int = -1) -> np.ndarray:
    """Compute a numerically stable softmax competition distribution along one axis."""
    if temperature <= 0.0 or not math.isfinite(float(temperature)):
        raise ValueError("temperature must be finite and positive")
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return np.zeros_like(array)
    logits = array / temperature
    logits = logits - np.max(logits, axis=axis, keepdims=True)
    exponentials = np.exp(np.clip(logits, -80.0, 0.0))
    return exponentials / np.maximum(np.sum(exponentials, axis=axis, keepdims=True), 1e-12)


def normalize_vector(values: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    """L2-normalize a vector while preserving an all-zero vector as neutral evidence."""
    vector = np.asarray(values, dtype=np.float64).ravel()
    norm = float(np.linalg.norm(vector))
    return vector / max(norm, epsilon)


def cosine_similarity(left: np.ndarray, right: np.ndarray, epsilon: float = 1e-8) -> float:
    """Calculate bounded cosine similarity for equal-dimensional finite vectors."""
    a = np.asarray(left, dtype=np.float64).ravel()
    b = np.asarray(right, dtype=np.float64).ravel()
    if a.shape != b.shape:
        raise ValueError("cosine similarity vectors must have matching dimensions")
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator <= epsilon:
        return 1.0 if np.allclose(a, b) else 0.0
    return float(np.clip(np.dot(a, b) / denominator, -1.0, 1.0))


def contrast_normalize(image: np.ndarray, sigma: float = 3.0, epsilon: float = 1e-8) -> dict[str, np.ndarray]:
    """Estimate local luminance/variance, subtract local mean, and divide by local contrast scale."""
    gray = as_gray(image)
    local_mean = ndi.gaussian_filter(gray, sigma=sigma, mode="reflect")
    local_second = ndi.gaussian_filter(gray * gray, sigma=sigma, mode="reflect")
    local_variance = np.maximum(local_second - local_mean * local_mean, 0.0)
    local_contrast = gray - local_mean
    normalized = local_contrast / np.sqrt(local_variance + epsilon)
    return {
        "grayscale": gray,
        "local_luminance": local_mean,
        "local_variance": local_variance,
        "local_contrast": local_contrast,
        "normalized_image": normalized,
    }


def gabor_kernel(
    size: int, orientation: float, spatial_frequency: float, sigma: float, gamma: float, phase: float,
) -> np.ndarray:
    """Construct a zero-mean, unit-L2 Gabor kernel for one orientation/frequency/phase channel."""
    coordinates = np.arange(size, dtype=np.float64) - (size - 1.0) / 2.0
    yy, xx = np.meshgrid(coordinates, coordinates, indexing="ij")
    rotated_x = xx * math.cos(orientation) + yy * math.sin(orientation)
    rotated_y = -xx * math.sin(orientation) + yy * math.cos(orientation)
    envelope = np.exp(-(rotated_x**2 + gamma**2 * rotated_y**2) / (2.0 * sigma**2))
    carrier = np.cos(2.0 * math.pi * spatial_frequency * rotated_x + phase)
    kernel = envelope * carrier
    kernel -= float(np.mean(kernel))
    norm = float(np.linalg.norm(kernel))
    return kernel / max(norm, 1e-12)


def gabor_energy_bank(
    image: np.ndarray,
    orientations: Sequence[float],
    spatial_frequencies: Sequence[float],
    size: int,
    sigma: float,
    gamma: float,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Convolve even/odd Gabor quadrature pairs and return orientation-frequency energy maps."""
    plane = np.asarray(image, dtype=np.float64)
    if plane.ndim != 2:
        raise ValueError("Gabor input must be a two-dimensional image")
    even_maps: list[np.ndarray] = []
    odd_maps: list[np.ndarray] = []
    energy_maps: list[np.ndarray] = []
    for orientation in orientations:
        for frequency in spatial_frequencies:
            even = gabor_kernel(size, float(orientation), float(frequency), sigma, gamma, 0.0)
            odd = gabor_kernel(size, float(orientation), float(frequency), sigma, gamma, math.pi / 2.0)
            response_even = ndi.convolve(plane, even, mode="reflect")
            response_odd = ndi.convolve(plane, odd, mode="reflect")
            even_maps.append(response_even)
            odd_maps.append(response_odd)
            energy_maps.append(np.sqrt(response_even**2 + response_odd**2 + 1e-12))
    stack = np.asarray(energy_maps, dtype=np.float64)
    return stack, {
        "even_responses": np.asarray(even_maps, dtype=np.float64),
        "odd_responses": np.asarray(odd_maps, dtype=np.float64),
        "energy_maps": stack,
    }


def divisive_normalize(features: np.ndarray, epsilon: float = 1e-8, gain: float = 0.5) -> np.ndarray:
    """Normalize channel activity by pooled local energy to model gain control and competition."""
    array = np.maximum(np.asarray(features, dtype=np.float64), 0.0)
    pooled = np.sum(array**2, axis=0, keepdims=True)
    local_pool = ndi.gaussian_filter(pooled, sigma=(0.0, 1.5, 1.5), mode="reflect")
    denominator = np.sqrt(epsilon + gain * local_pool)
    return array / denominator


def connected_regions(
    feature_image: np.ndarray, max_regions: int = 128, minimum_area: int | None = None,
) -> list[dict[str, object]]:
    """Find high-contrast connected components and preserve their bounding boxes and centroids."""
    image = np.asarray(feature_image, dtype=np.float64)
    if image.ndim != 2:
        raise ValueError("feature_image must be two-dimensional")
    strength = np.abs(image)
    positive = strength[strength > 1e-9]
    if positive.size == 0:
        return []
    threshold = float(np.quantile(positive, 0.72))
    mask = strength >= threshold
    labels, count = ndi.label(mask)
    objects = ndi.find_objects(labels)
    area_floor = minimum_area if minimum_area is not None else max(2, int(image.size * 0.0002))
    regions: list[dict[str, object]] = []
    for label, slices in enumerate(objects, start=1):
        if slices is None:
            continue
        local_mask = labels[slices] == label
        area = int(np.count_nonzero(local_mask))
        if area < area_floor:
            continue
        y_slice, x_slice = slices
        ys, xs = np.nonzero(local_mask)
        y0, x0 = int(y_slice.start), int(x_slice.start)
        regions.append({
            "region_id": f"r{label:03d}",
            "bbox": (y0, x0, int(y_slice.stop), int(x_slice.stop)),
            "centroid_yx": (float(y0 + np.mean(ys)), float(x0 + np.mean(xs))),
            "area": area,
            "mask": local_mask,
        })
    regions.sort(key=lambda item: (item["bbox"][1], item["bbox"][0]))
    return regions[:max_regions]


def pool_region_descriptor(feature_stack: np.ndarray, region: Mapping[str, object], grid_size: int = 4) -> np.ndarray:
    """Pool each feature channel over a fixed region grid and append stable summary statistics."""
    stack = np.asarray(feature_stack, dtype=np.float64)
    y0, x0, y1, x1 = region["bbox"]
    crop = stack[:, int(y0):int(y1), int(x0):int(x1)]
    if crop.size == 0:
        return np.zeros(stack.shape[0] * grid_size * grid_size + stack.shape[0] * 2, dtype=np.float64)
    pooled = []
    for channel in crop:
        resized = ndi.zoom(channel, (grid_size / max(channel.shape[0], 1), grid_size / max(channel.shape[1], 1)), order=1)
        resized = resized[:grid_size, :grid_size]
        if resized.shape != (grid_size, grid_size):
            fixed = np.zeros((grid_size, grid_size), dtype=np.float64)
            fixed[:resized.shape[0], :resized.shape[1]] = resized
            resized = fixed
        pooled.extend(resized.ravel().tolist())
    statistics = np.concatenate((np.mean(crop, axis=(1, 2)), np.std(crop, axis=(1, 2))))
    return np.concatenate((np.asarray(pooled, dtype=np.float64), statistics))


def normalize_orthography(value: str) -> str:
    """Apply Unicode compatibility normalization and case folding while preserving written order."""
    return unicodedata.normalize("NFKC", str(value)).casefold().strip()


def ordered_edit_similarity(left: str, right: str) -> float:
    """Calculate normalized Levenshtein similarity while penalizing letter substitutions and reordering."""
    a, b = normalize_orthography(left), normalize_orthography(right)
    if not a and not b:
        return 1.0
    previous = list(range(len(b) + 1))
    for index_a, char_a in enumerate(a, start=1):
        current = [index_a]
        for index_b, char_b in enumerate(b, start=1):
            substitution = 0 if char_a == char_b else 1
            current.append(min(
                current[-1] + 1,
                previous[index_b] + 1,
                previous[index_b - 1] + substitution,
            ))
        previous = current
    return float(1.0 - previous[-1] / max(len(a), len(b), 1))


def extract_ordered_ngrams(sequence: Sequence[str], sizes: Sequence[int] = (2, 3, 4)) -> list[dict[str, object]]:
    """Extract overlapping position-tagged n-grams and retain ordered non-adjacent character pairs."""
    items: list[dict[str, object]] = []
    for size in sizes:
        for start in range(max(0, len(sequence) - size + 1)):
            items.append({"n": int(size), "start": start, "unit": "".join(sequence[start:start + size])})
    return items


def morphology_segmentations(
    word: str, inventory: Sequence[object], max_depth: int = 4, limit: int = 64,
) -> list[dict[str, object]]:
    """Enumerate bounded concatenative morpheme analyses using only entries in the supplied inventory."""
    normalized = normalize_orthography(word)
    entries = [item for item in inventory if normalize_orthography(item.form)]
    by_first = sorted(entries, key=lambda item: (-len(normalize_orthography(item.form)), item.form))
    results: list[dict[str, object]] = []

    def visit(offset: int, path: list[object]) -> None:
        """Explore one bounded segmentation path and retain only complete decompositions."""
        if len(results) >= limit:
            return
        if offset == len(normalized):
            if len(path) >= 2:
                results.append({
                    "morphemes": [item.form for item in path],
                    "roles": [item.role for item in path],
                    "frequency_score": float(sum(math.log1p(float(item.frequency)) for item in path)),
                    "source_references": [item.source_reference for item in path],
                    "semantic_features": [dict(item.semantic_features) for item in path],
                    "grammatical_features": [dict(item.grammatical_features) for item in path],
                })
            return
        if len(path) >= max_depth:
            return
        for entry in by_first:
            form = normalize_orthography(entry.form)
            if normalized.startswith(form, offset):
                visit(offset + len(form), path + [entry])

    visit(0, [])
    results.sort(key=lambda item: float(item["frequency_score"]), reverse=True)
    return results


def safe_array_summary(values: np.ndarray) -> dict[str, object]:
    """Return finite scalar diagnostics for an array without serializing all of its elements."""
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"shape": list(array.shape), "size": 0, "finite": True}
    return {
        "shape": [int(item) for item in array.shape],
        "size": int(array.size),
        "finite": bool(np.all(np.isfinite(array))),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
        "mean": float(np.mean(array)),
        "standard_deviation": float(np.std(array)),
    }
