"""Archive cell indexing over discretized behavior descriptors (steps 39-40)."""
from __future__ import annotations

from typing import Any, Iterable, Mapping

from ..domain.behavior_descriptor import (
    DESCRIPTOR_DIMENSIONS, BehaviorDescriptor, DescriptorCalibration,
)


class CellIndex:
    """Maps behavior descriptors to archive cells using versioned boundaries (step 40).

    Cell identity is the discretized coordinate tuple plus the calibration version,
    so a recalibration (new boundaries) never silently reuses old cells — the archive
    version changes with it (acceptance test 30: descriptor boundaries and archive
    version are reproducible).
    """

    def __init__(self, calibration: DescriptorCalibration) -> None:
        self.calibration = calibration
        self._dimensions = {dim.name: dim for dim in calibration.dimensions}
        if set(self._dimensions) != set(DESCRIPTOR_DIMENSIONS):
            raise ValueError("calibration must cover every descriptor dimension")

    @property
    def version(self) -> str:
        """Archive version — derived from the calibration version (step 40)."""
        return self.calibration.version

    def total_cells(self) -> int:
        """Total number of cells in the discretized descriptor space."""
        total = 1
        for dimension in self.calibration.dimensions:
            total *= dimension.bins
        return total

    def cell_id(self, descriptor: BehaviorDescriptor) -> tuple[tuple[int, ...], str] | None:
        """Return ``(coordinates, cell_id)`` or ``None`` when binning fails.

        Missing or out-of-range measurements are handled explicitly: the descriptor
        is rejected (step 40) instead of being silently clipped into a cell.
        """
        coordinates, problems = descriptor.cell_coordinates(self.calibration)
        if any(coordinate is None for coordinate in coordinates) or problems:
            return None
        return coordinates, self.encode(coordinates)

    def encode(self, coordinates: tuple[int, ...]) -> str:
        """Stable cell identifier from discretized coordinates + version."""
        return f"{self.version}:{'|'.join(str(c) for c in coordinates)}"

    def boundaries(self) -> dict[str, list[float]]:
        """Versioned bin boundaries (reproducibility evidence, step 40)."""
        return self.calibration.boundaries()

    def descriptor_is_in_range(self, descriptor: BehaviorDescriptor) -> tuple[bool, tuple[str, ...]]:
        """Explicit in-range check with reasons for missing/out-of-range values."""
        _coordinates, problems = descriptor.cell_coordinates(self.calibration)
        return (not problems), tuple(problems)

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "total_cells": self.total_cells(),
            "boundaries": self.boundaries(),
        }


def neighboring_cells(
    coordinates: tuple[int, ...],
    bins_per_dimension: Iterable[int],
) -> list[tuple[int, ...]]:
    """Adjacent cells in the discretized grid (frontier bias for parent selection)."""
    neighbors: list[tuple[int, ...]] = []
    dims = list(bins_per_dimension)
    for index, bins in enumerate(dims):
        for delta in (-1, 1):
            neighbor = list(coordinates)
            neighbor[index] += delta
            if 0 <= neighbor[index] < bins:
                neighbors.append(tuple(neighbor))
    return neighbors


__all__ = ["CellIndex", "neighboring_cells"]
