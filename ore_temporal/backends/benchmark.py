"""Backend benchmarking before adoption (specification Step 45)."""
from __future__ import annotations

from dataclasses import dataclass
import time as _time
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from ..monitoring import mean_absolute_error

__all__ = ("BackendBenchmark", "BenchmarkEntry", "benchmark_identical_workload")


@dataclass(frozen=True, slots=True)
class BenchmarkEntry:
    """Measured metrics for one backend on the identical workload."""

    backend: str
    latency_ms: float
    throughput_per_second: float
    accuracy_metric: float | None
    energy_note: str = "energy not measurable in the CPU simulation environment"
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize the benchmark entry."""
        return {
            "backend": self.backend,
            "latency_ms": self.latency_ms,
            "throughput_per_second": self.throughput_per_second,
            "accuracy_metric": self.accuracy_metric,
            "energy_note": self.energy_note,
            "notes": self.notes,
        }


class BackendBenchmark:
    """Compare identical workloads across backends with wall-clock timing."""

    def __init__(self, repeat: int = 3) -> None:
        """Configure the number of timing repetitions."""
        if repeat <= 0:
            raise ValueError("repeat must be positive")
        self.repeat = int(repeat)

    def run(self, backend_name: str, workload: Callable[[], Any],
            observed: Sequence[float] | None = None,
            predictions_of: Callable[[Any], Sequence[float]] | None = None,
            ) -> BenchmarkEntry:
        """Time one backend on the workload and compute accuracy when labels exist."""
        timings: list[float] = []
        result: Any = None
        for _ in range(self.repeat):
            started = _time.perf_counter()
            result = workload()
            timings.append((_time.perf_counter() - started) * 1000.0)
        latency = float(np.median(timings))
        throughput = 1.0 / (latency / 1000.0) if latency > 0 else float("inf")
        accuracy = None
        if observed is not None and predictions_of is not None and result is not None:
            accuracy = mean_absolute_error(observed, predictions_of(result))
        return BackendEntryBuilder(backend_name, latency, throughput, accuracy).build()


class BackendEntryBuilder:
    """Assemble a BenchmarkEntry from measured quantities."""

    def __init__(self, backend_name: str, latency_ms: float, throughput: float,
                 accuracy: float | None) -> None:
        """Store the measured quantities."""
        self.backend_name = backend_name
        self.latency_ms = latency_ms
        self.throughput = throughput
        self.accuracy = accuracy

    def build(self, notes: str = "") -> BenchmarkEntry:
        """Materialize the entry."""
        return BenchmarkEntry(
            backend=self.backend_name,
            latency_ms=round(self.latency_ms, 3),
            throughput_per_second=round(self.throughput, 3),
            accuracy_metric=(None if self.accuracy is None else round(self.accuracy, 6)),
            notes=notes)


def benchmark_identical_workload(workloads: Mapping[str, Callable[[], Any]],
                                 repeat: int = 3,
                                 observed: Sequence[float] | None = None,
                                 predictions_of: Callable[[Any], Sequence[float]] | None = None,
                                 ) -> dict[str, Any]:
    """Benchmark identical windows across named backends (Step 45)."""
    benchmark = BackendBenchmark(repeat=repeat)
    entries = {name: benchmark.run(name, workload, observed, predictions_of)
               for name, workload in workloads.items()}
    return {
        "entries": {name: entry.to_dict() for name, entry in entries.items()},
        "identical_workload": True,
        "repeat": repeat,
        "hardware_adoption": "separate decision; software benchmark only",
    }
