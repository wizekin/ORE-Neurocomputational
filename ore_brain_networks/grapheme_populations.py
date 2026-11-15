"""Operation 2: spatially ordered grapheme regions and neural population-style prototypes."""
from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy import ndimage as ndi

from .contracts import BNBEAConfig, GraphemePrototype
from .visual_gabor import extract_region_feature_vector


def segment_candidate_regions(
    feature_stack: np.ndarray, quantile: float = 0.72, max_regions: int = 128
) -> list[dict[str, object]]:
    """Threshold combined edge energy and return connected candidate character regions in reading order."""
    if feature_stack.ndim != 3:
        raise ValueError("feature_stack must have shape [channels, height, width]")
    energy = np.max(feature_stack, axis=0)
    threshold = float(np.quantile(energy, quantile))
    foreground = energy > threshold
    foreground = ndi.binary_closing(foreground, structure=np.ones((2, 2), dtype=bool))
    labels, count = ndi.label(foreground)
    objects = ndi.find_objects(labels)
    candidates: list[dict[str, object]] = []
    for label_id, slices in enumerate(objects, start=1):
        if slices is None:
            continue
        y_slice, x_slice = slices
        height = y_slice.stop - y_slice.start
        width = x_slice.stop - x_slice.start
        if height < 2 or width < 2:
            continue
        mask = labels == label_id
        candidates.append({
            "region_id": f"region-{label_id:04d}",
            "bbox": (int(x_slice.start), int(y_slice.start), int(x_slice.stop), int(y_slice.stop)),
            "mask": mask,
            "feature_vector": extract_region_feature_vector(feature_stack, mask),
            "area": int(np.count_nonzero(mask)),
        })
    candidates.sort(key=lambda item: (item["bbox"][1], item["bbox"][0]))
    return candidates[:max_regions]


def _softmax(scores: np.ndarray, temperature: float) -> np.ndarray:
    """Convert unnormalized population scores into a stable probability distribution."""
    scaled = scores / max(float(temperature), 1e-9)
    shifted = scaled - np.max(scaled)
    values = np.exp(np.clip(shifted, -80.0, 0.0))
    return values / max(float(np.sum(values)), 1e-12)


class GraphemePopulationDecoder:
    """Decode candidate visual regions using multiple feature-tuned population prototypes."""

    def __init__(self, config: BNBEAConfig | None = None) -> None:
        """Store temperature and segmentation settings for population-style grapheme competition."""
        self.config = config or BNBEAConfig()

    def process(
        self, feature_stack: np.ndarray, prototypes: Sequence[GraphemePrototype] = ()
    ) -> dict[str, object]:
        """Segment features, compute prototype distances, and retain candidate order and uncertainty."""
        regions = segment_candidate_regions(
            feature_stack,
            self.config.visual_threshold_quantile,
            self.config.max_candidate_regions,
        )
        decoded: list[dict[str, object]] = []
        for region in regions:
            vector = np.asarray(region["feature_vector"], dtype=np.float64)
            activations: dict[str, float] = {}
            if prototypes:
                scores: list[float] = []
                labels: list[str] = []
                for prototype in prototypes:
                    mean = np.asarray(prototype.mean, dtype=np.float64)
                    variance = np.asarray(prototype.variance, dtype=np.float64)
                    if mean.shape != vector.shape:
                        raise ValueError(
                            f"prototype {prototype.label!r} has {mean.size} features; "
                            f"region extractor produced {vector.size}"
                        )
                    delta = vector - mean
                    distance_squared = float(np.sum((delta**2) / variance))
                    scores.append(-0.5 * distance_squared)
                    labels.append(prototype.label)
                probabilities = _softmax(np.asarray(scores), self.config.population_temperature)
                for label, probability in zip(labels, probabilities):
                    activations[label] = float(probability)
                ranked = sorted(activations.items(), key=lambda pair: pair[1], reverse=True)
                decoded.append({
                    **{key: value for key, value in region.items() if key != "mask"},
                    "population_activations": activations,
                    "ranked_candidates": ranked,
                    "best_candidate": ranked[0][0] if ranked else None,
                    "confidence": float(ranked[0][1]) if ranked else 0.0,
                })
            else:
                decoded.append({
                    **{key: value for key, value in region.items() if key != "mask"},
                    "population_activations": {},
                    "ranked_candidates": [],
                    "best_candidate": None,
                    "confidence": 0.0,
                })
        sequence = [item["best_candidate"] for item in decoded if item["best_candidate"] is not None]
        return {
            "candidate_regions": decoded,
            "position_preserving_sequence": [
                {"position": index, "grapheme": label} for index, label in enumerate(sequence)
            ],
            "prototype_count": len(prototypes),
            "recognition_status": "recognized_with_supplied_prototypes" if prototypes else "features_only_no_prototypes",
            "biological_analogue": (
                "Distributed grapheme-sensitive population responses with competing tuning curves; "
                "the prototype bank is a transparent engineering construct."
            ),
        }

    @staticmethod
    def update_prototype(
        prototype: GraphemePrototype, observed_vector: np.ndarray, learning_rate: float = 0.05
    ) -> GraphemePrototype:
        """Return an exponential-moving-average prototype update from one labelled observation."""
        vector = np.asarray(observed_vector, dtype=np.float64)
        mean = np.asarray(prototype.mean, dtype=np.float64)
        if vector.shape != mean.shape:
            raise ValueError("observed_vector must match prototype feature dimensions")
        rate = float(learning_rate)
        if not 0.0 <= rate <= 1.0:
            raise ValueError("learning_rate must lie in [0, 1]")
        updated = (1.0 - rate) * mean + rate * vector
        residual = (vector - updated) ** 2
        variance = (1.0 - rate) * np.asarray(prototype.variance) + rate * np.maximum(residual, 1e-4)
        return GraphemePrototype(prototype.label, updated, variance, prototype.prototype_id)
