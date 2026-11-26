"""Behavior descriptors and versioned calibration (sections 2.4, steps 39-40).

A behavior descriptor (section 2.4) is ``b(s) = (speed, reasoning depth, resource
tier, reliability, module cooperation)`` discretized into archive cells. Descriptor
normalisation (step 40) fixes ranges and bin boundaries from a *versioned calibration
set*; missing or out-of-range measurements are handled explicitly by policy rather
than silently binned.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .genome import canonical_json, content_digest

#: Descriptor dimension names in canonical order (step 39).
DESCRIPTOR_DIMENSIONS: tuple[str, ...] = (
    "speed", "reasoning_depth", "resource_tier", "reliability", "module_cooperation",
)

#: Missing-value policy: reject candidates whose descriptors cannot be binned.
MISSING_VALUE_POLICY = "reject"
OUT_OF_RANGE_POLICY = "reject"


@dataclass(frozen=True, slots=True)
class DescriptorDimension:
    """One discretized descriptor dimension with fixed boundaries (step 40)."""

    name: str
    low: float
    high: float
    bins: int

    def __post_init__(self) -> None:
        if not (self.low < self.high):
            raise ValueError(f"dimension {self.name!r} needs low < high")
        if self.bins < 1:
            raise ValueError(f"dimension {self.name!r} needs at least one bin")

    @property
    def width(self) -> float:
        return self.high - self.low

    def normalize(self, value: float) -> float:
        return (value - self.low) / self.width

    def discretize(self, value: float | None) -> tuple[int | None, str | None]:
        """Bin one value; missing/out-of-range measurements are explicit (step 40)."""
        if value is None or (isinstance(value, float) and not math.isfinite(value)):
            return None, "missing measurement"
        if value < self.low or value > self.high:
            return None, f"out-of-range measurement {value:.4g} not in [{self.low:.4g}, {self.high:.4g}]"
        position = (min(max(value, self.low), self.high) - self.low) / self.width
        bin_index = min(int(position * self.bins), self.bins - 1)
        return bin_index, None

    def boundary_list(self) -> list[float]:
        """Reproducible bin boundaries for this dimension (acceptance test 30)."""
        return [round(self.low + (i / self.bins) * self.width, 6) for i in range(self.bins + 1)]

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "low": self.low, "high": self.high, "bins": self.bins}


class DescriptorCalibration:
    """Versioned calibration set fixing ranges and bin boundaries (step 40).

    The calibration set is the immutable reference used to derive dimension ranges;
    a changed calibration set produces a new archive version so cell boundaries stay
    reproducible (acceptance test 30: descriptor boundaries/archive version are
    reproducible).
    """

    def __init__(
        self,
        dimensions: tuple[DescriptorDimension, ...],
        *,
        version: str,
        source_digest: str,
    ) -> None:
        names = [dimension.name for dimension in dimensions]
        if tuple(names) != DESCRIPTOR_DIMENSIONS:
            raise ValueError(
                "descriptor dimensions must be exactly "
                f"{DESCRIPTOR_DIMENSIONS}, got {tuple(names)}"
            )
        self.dimensions = dimensions
        self.version = version
        self.source_digest = source_digest

    @classmethod
    def from_samples(
        cls,
        samples: Sequence[Mapping[str, float]],
        *,
        version: str,
        bins: int = 4,
        padding: float = 0.05,
    ) -> "DescriptorCalibration":
        """Derive fixed ranges from a versioned calibration set of raw measurements.

        Missing dimensions in samples are ignored for range derivation; the digest of
        the full sample set is recorded so recalibration is detectable.
        """
        dimensions: list[DescriptorDimension] = []
        for name in DESCRIPTOR_DIMENSIONS:
            values = [float(sample[name]) for sample in samples if name in sample and sample[name] is not None]
            values = [v for v in values if math.isfinite(v)]
            if not values:
                raise ValueError(f"calibration set has no usable samples for {name!r}")
            low, high = min(values), max(values)
            span = max(high - low, 1e-9)
            low, high = low - padding * span, high + padding * span
            dimensions.append(DescriptorDimension(name, round(low, 6), round(high, 6), bins))
        return cls(
            tuple(dimensions),
            version=version,
            source_digest=content_digest({
                "samples": [dict(sorted(dict(s).items())) for s in samples],
            }),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "source_digest": self.source_digest,
            "dimensions": [dimension.as_dict() for dimension in self.dimensions],
        }

    def boundaries(self) -> dict[str, list[float]]:
        """All bin boundaries, keyed by dimension (reproducibility evidence)."""
        return {dimension.name: dimension.boundary_list() for dimension in self.dimensions}


#: Reference measurements used by default calibrations (a fixed, versioned set).
DEFAULT_CALIBRATION_SAMPLES: tuple[dict[str, float], ...] = tuple(
    {
        "speed": speed,
        "reasoning_depth": depth,
        "resource_tier": resource,
        "reliability": reliability,
        "module_cooperation": cooperation,
    }
    for speed, depth, resource, reliability, cooperation in (
        # (latency ms, module activations, cost per task, success rate, cooperation)
        (120.0, 2.0, 6.0, 0.60, 0.30),
        (180.0, 3.0, 8.0, 0.80, 0.55),
        (320.0, 5.0, 14.0, 0.90, 0.70),
        (450.0, 7.0, 20.0, 0.75, 0.45),
        (260.0, 4.0, 11.0, 0.85, 0.65),
        (520.0, 8.0, 28.0, 0.70, 0.40),
        (390.0, 6.0, 17.0, 0.95, 0.80),
        (240.0, 4.5, 10.0, 0.88, 0.60),
        (600.0, 9.0, 48.0, 0.65, 0.35),
        (150.0, 3.5, 9.0, 1.00, 1.00),
        (700.0, 10.0, 55.0, 0.60, 0.30),
    )
)


def default_calibration(version: str = "calibration-v1") -> DescriptorCalibration:
    """The default versioned calibration used by the evolution service."""
    return DescriptorCalibration.from_samples(DEFAULT_CALIBRATION_SAMPLES, version=version)


@dataclass(frozen=True, slots=True)
class BehaviorDescriptor:
    """One measured behavior descriptor with its calibration version (step 39)."""

    values: Mapping[str, float | None]
    calibration_version: str

    def cell_coordinates(
        self, calibration: DescriptorCalibration,
    ) -> tuple[tuple[int | None, ...], list[str]]:
        """Discretize into archive-cell coordinates; returns (coords, problems)."""
        coords: list[int | None] = []
        problems: list[str] = []
        for dimension in calibration.dimensions:
            value = self.values.get(dimension.name)
            bin_index, problem = dimension.discretize(value)
            coords.append(bin_index)
            if problem is not None:
                problems.append(f"{dimension.name}: {problem}")
        return tuple(coords), problems

    def distance(self, other: "BehaviorDescriptor") -> float:
        """Normalized Euclidean distance over shared dimensions (novelty, 2.5)."""
        total = 0.0
        count = 0
        for name in DESCRIPTOR_DIMENSIONS:
            a, b = self.values.get(name), other.values.get(name)
            if a is None or b is None:
                continue
            total += (float(a) - float(b)) ** 2
            count += 1
        if count == 0:
            return 0.0
        return math.sqrt(total / count)

    def as_dict(self) -> dict[str, Any]:
        return {
            "calibration_version": self.calibration_version,
            "values": {name: self.values.get(name) for name in DESCRIPTOR_DIMENSIONS},
        }

    @classmethod
    def from_evaluation(cls, measurement: Mapping[str, float | None], calibration_version: str) -> "BehaviorDescriptor":
        """Build a descriptor from evaluation aggregates (step 39)."""
        values = {name: measurement.get(name) for name in DESCRIPTOR_DIMENSIONS}
        return cls(values, calibration_version)

    def content_payload(self) -> dict[str, Any]:
        return {
            "calibration_version": self.calibration_version,
            "values": {k: self.values.get(k) for k in DESCRIPTOR_DIMENSIONS},
        }

    def digest(self) -> str:
        return content_digest(self.content_payload())

    def normalized_vector(self, calibration: DescriptorCalibration) -> list[float]:
        """Descriptor values normalized into [0,1] using the calibration ranges."""
        normalized: list[float] = []
        for dimension in calibration.dimensions:
            value = self.values.get(dimension.name)
            if value is None:
                normalized.append(0.0)
            else:
                normalized.append(min(max(dimension.normalize(float(value)), 0.0), 1.0))
        return normalized


__all__ = [
    "DEFAULT_CALIBRATION_SAMPLES", "DESCRIPTOR_DIMENSIONS", "MISSING_VALUE_POLICY",
    "BehaviorDescriptor", "DescriptorCalibration", "DescriptorDimension",
    "default_calibration",
]
