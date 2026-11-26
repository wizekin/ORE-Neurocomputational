"""Edge specialist compression with measured trade-offs (Point 14, Step 39).

Quantization, pruning, and distillation are evaluated against the uncompressed
reference; numerical deviation and task-quality loss are measured together
(retention Q_compressed / Q_reference), and the uncompressed reference is preserved
so rollback is always possible.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

from ..config import LearningConfig
from ..functional import quantization_error, quantize_weights

__all__ = ("CompressionReport", "CompressionPlan", "compress_for_edge",
           "prune_weights", "pruning_sparsity")


def prune_weights(weights: dict[str, np.ndarray], sparsity: float
                  ) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """Magnitude-prune the smallest-magnitude entries to the target sparsity."""
    if not 0.0 <= sparsity < 1.0:
        raise ValueError("sparsity must be in [0, 1)")
    pruned: dict[str, np.ndarray] = {}
    removed: dict[str, int] = {}
    for name, array in weights.items():
        values = np.asarray(array, dtype=float)
        threshold = np.quantile(np.abs(values).ravel(), sparsity) \
            if values.size and sparsity > 0 else 0.0
        kept = np.where(np.abs(values) >= threshold, values, 0.0)
        pruned[name] = kept
        removed[name] = int(np.sum(kept == 0.0))
    return pruned, removed


def pruning_sparsity(weights: dict[str, np.ndarray]) -> float:
    """The achieved zero-share of a pruned weight dictionary."""
    total = sum(np.asarray(array).size for array in weights.values())
    zeros = sum(int(np.sum(np.asarray(array) == 0.0))
                for array in weights.values())
    return zeros / max(total, 1)


@dataclass(slots=True)
class CompressionReport:
    """Step 39 -- measured trade-offs of one compression recipe."""

    method: str
    quality_compressed: float
    quality_reference: float
    retention: float
    memory_compressed_bytes: int
    memory_reference_bytes: int
    memory_reduction: float
    numerical_deviation: float
    rollback_reference_preserved: bool
    within_thresholds: bool

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe view."""
        return {"method": self.method,
                "quality_compressed": round(self.quality_compressed, 6),
                "quality_reference": round(self.quality_reference, 6),
                "retention": round(self.retention, 6),
                "memory_compressed_bytes": self.memory_compressed_bytes,
                "memory_reference_bytes": self.memory_reference_bytes,
                "memory_reduction": round(self.memory_reduction, 6),
                "numerical_deviation": round(self.numerical_deviation, 9),
                "rollback_reference_preserved": self.rollback_reference_preserved,
                "within_thresholds": self.within_thresholds,
                "note": "the uncompressed reference stays available for rollback"}


@dataclass(slots=True)
class CompressionPlan:
    """A configured compression recipe (quantization, pruning, or both)."""

    bits: int = 8
    sparsity: float = 0.0

    def describe(self) -> dict[str, Any]:
        """JSON-safe view."""
        return {"bits": self.bits, "sparsity": self.sparsity}


def compress_for_edge(model: Any, quality_fn: Callable[[Any], float], *,
                      plan: CompressionPlan | None = None,
                      config: LearningConfig | None = None,
                      clone_fn: Callable[[Any], Any] | None = None
                      ) -> tuple[Any, CompressionReport]:
    """Quantize (and optionally prune) a model copy; measure retention vs reference.

    ``quality_fn(model)`` returns a higher-is-better task quality in [0, 1].  The
    original model object is never modified -- the compressed variant runs on a copy,
    and the reference is preserved for device-specific rollback (Step 39).
    """
    effective = config or LearningConfig()
    recipe = plan or CompressionPlan()
    reference_quality = float(quality_fn(model))
    reference_bytes = model.memory_bytes()

    work_model = (clone_fn(model) if clone_fn is not None else model.copy())
    if recipe.sparsity > 0.0:
        pruned, removed = prune_weights(work_model.parameters(), recipe.sparsity)
        work_model.set_parameters(pruned)
    quantized, scales, dequantized = quantize_weights(work_model.parameters(),
                                                      bits=recipe.bits)
    deviation = quantization_error(work_model.parameters(), dequantized)
    work_model.set_parameters(dequantized)
    compressed_quality = float(quality_fn(work_model))
    compressed_bytes = sum(array.size for array in quantized.values()) * (
        recipe.bits // 8)

    retention = (compressed_quality / reference_quality
                 if reference_quality != 0.0 else 0.0)
    method = (f"quantization(bits={recipe.bits})"
              + (f"+pruning(sparsity={recipe.sparsity:.2f})"
                 if recipe.sparsity > 0 else ""))
    within = (retention >= effective.edge_retention_threshold
              and compressed_bytes / 1e6 <= effective.edge_max_memory_mb)
    report = CompressionReport(
        method=method, quality_compressed=compressed_quality,
        quality_reference=reference_quality, retention=float(retention),
        memory_compressed_bytes=int(compressed_bytes),
        memory_reference_bytes=int(reference_bytes),
        memory_reduction=float(1.0 - compressed_bytes / max(reference_bytes, 1)),
        numerical_deviation=float(deviation),
        rollback_reference_preserved=True, within_thresholds=bool(within))
    return work_model, report
