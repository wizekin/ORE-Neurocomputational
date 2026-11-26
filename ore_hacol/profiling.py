"""Workload profiling: benchmark fixtures, cost measurement, bottleneck
classification, and optimization-value estimation (integrated steps 5-8).

All fixtures are seeded and reproducible; synthetic data is used where
production data cannot be retained (step 5). Measurements separate kernel
time from orchestration (step 6) and classify the bottleneck with a simple
measured roofline argument (step 7).
"""
from __future__ import annotations

import statistics
import time
import tracemalloc
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .contracts import HardwareProfile, ValueEstimate


@dataclass(frozen=True)
class LatencyStats:
    """Median/p95/p99 latency report (Phase G requirement)."""

    count: int
    median: float
    p95: float
    p99: float
    mean: float
    minimum: float
    maximum: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "median_seconds": self.median,
            "p95_seconds": self.p95,
            "p99_seconds": self.p99,
            "mean_seconds": self.mean,
            "min_seconds": self.minimum,
            "max_seconds": self.maximum,
        }


@dataclass
class ProfileReport:
    """Measured cost evidence for one workload (integrated step 6)."""

    name: str
    latency: LatencyStats
    warmup_seconds: float
    throughput_per_second: float
    kernel_seconds: float
    orchestration_seconds: float
    peak_memory_bytes: int
    allocations: int
    flops: float
    bytes_moved: float
    arithmetic_intensity: float
    bottleneck_class: str
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "latency": self.latency.to_dict(),
            "warmup_seconds": self.warmup_seconds,
            "throughput_per_second": self.throughput_per_second,
            "kernel_seconds": self.kernel_seconds,
            "orchestration_seconds": self.orchestration_seconds,
            "peak_memory_bytes": self.peak_memory_bytes,
            "allocations": self.allocations,
            "flops": self.flops,
            "bytes_moved": self.bytes_moved,
            "arithmetic_intensity": self.arithmetic_intensity,
            "bottleneck_class": self.bottleneck_class,
            "notes": list(self.notes),
        }


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    index = min(len(ordered) - 1, max(0, int(round(quantile * (len(ordered) - 1)))))
    return ordered[index]


def measure_latency(
    fn: Callable[[], Any],
    *,
    runs: int = 7,
    warmup: int = 1,
) -> tuple[LatencyStats, float, int, int, float]:
    """Time a callable with warmup; returns (stats, warmup seconds, allocs, peak bytes, total)."""
    warmup_start = time.perf_counter()
    for _ in range(max(0, warmup)):
        fn()
    warmup_seconds = time.perf_counter() - warmup_start

    tracemalloc.start()
    times: list[float] = []
    for _ in range(max(1, runs)):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    _, peak = tracemalloc.get_traced_memory()
    stats = tracemalloc.take_snapshot()
    allocations = len(stats.statistics("lineno"))
    tracemalloc.stop()

    latency = LatencyStats(
        count=len(times),
        median=statistics.median(times),
        p95=_percentile(times, 0.95),
        p99=_percentile(times, 0.99),
        mean=float(statistics.fmean(times)),
        minimum=min(times),
        maximum=max(times),
    )
    return latency, warmup_seconds, allocations, int(peak), float(sum(times))


def profile_function(
    fn: Callable[[], Any],
    name: str,
    *,
    flops: float = 0.0,
    bytes_moved: float = 0.0,
    runs: int = 7,
    overhead_seconds: float = 0.0,
) -> ProfileReport:
    """Measure end-to-end cost of one callable and separate kernel from orchestration."""
    latency, warmup_seconds, allocations, peak_bytes, total = measure_latency(fn, runs=runs)
    orchestration = overhead_seconds if overhead_seconds > 0 else max(
        0.0, latency.median * 0.0
    )
    kernel = max(0.0, latency.median - orchestration)
    intensity = (flops / bytes_moved) if bytes_moved > 0 else float("inf") if flops else 0.0
    throughput = (1.0 / latency.median) if latency.median > 0 else float("inf")
    return ProfileReport(
        name=name,
        latency=latency,
        warmup_seconds=warmup_seconds,
        throughput_per_second=throughput,
        kernel_seconds=kernel,
        orchestration_seconds=orchestration,
        peak_memory_bytes=peak_bytes,
        allocations=allocations,
        flops=flops,
        bytes_moved=bytes_moved,
        arithmetic_intensity=intensity if intensity == intensity else 0.0,
        bottleneck_class="unclassified",
        notes=["kernel vs orchestration separation requires an overhead estimate"],
    )


def classify_bottleneck(report: ProfileReport, profile: HardwareProfile) -> str:
    """Classify the bottleneck with measured rates (integrated step 7).

    Uses a roofline-style argument: predicted compute time vs predicted memory
    time from measured rates; when neither dominates, the workload is
    orchestration/latency bound.
    """
    if report.flops <= 0.0 and report.bytes_moved <= 0.0:
        return "latency_bound"
    compute_time = (
        report.flops / profile.compute_rate_flops_per_second
        if profile.compute_rate_flops_per_second > 0
        else float("inf")
    )
    memory_time = (
        report.bytes_moved / profile.bandwidth_bytes_per_second
        if profile.bandwidth_bytes_per_second > 0
        else float("inf")
    )
    if compute_time == memory_time == float("inf"):
        return "unclassified"
    total = compute_time + memory_time
    if total <= 0:
        return "unclassified"
    compute_share = compute_time / total
    if compute_share >= 0.7:
        return "compute_bound"
    if compute_share <= 0.3:
        return "bandwidth_bound"
    if report.orchestration_seconds > 0 and report.kernel_seconds <= 0:
        return "launch_or_overhead_bound"
    return "memory_bound" if memory_time >= compute_time else "mixed"


def synthetic_compute_bound_workload(size: int = 384, seed: int = 7) -> dict[str, Any]:
    """Dense GEMM probe used to calibrate/verify compute-bound classification."""
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((size, size))
    b = rng.standard_normal((size, size))
    return {
        "callable": lambda: a @ b,
        "flops": 2.0 * size**3,
        "bytes_moved": 3 * size * size * 8,
        "description": f"dense {size}x{size} float64 GEMM",
    }


def synthetic_bandwidth_bound_workload(elements: int = 1 << 20, seed: int = 8) -> dict[str, Any]:
    """Streaming copy probe used to calibrate/verify bandwidth-bound classification."""
    rng = np.random.default_rng(seed)
    source = rng.standard_normal(elements)
    destination = np.empty_like(source)
    return {
        "callable": lambda: destination.__setitem__(slice(None), source),
        "flops": 0.0,
        "bytes_moved": 2 * elements * 8,
        "description": f"elementwise copy of {elements} float64 values",
    }


# ---------------------------------------------------------------------------
# Representative benchmark fixtures (integrated step 5)
# ---------------------------------------------------------------------------


def attention_fixture(
    *,
    sequence: int = 96,
    dim: int = 16,
    value_dim: int | None = None,
    mask_kind: str = "none",
    seed: int = 21,
    extreme_logits: bool = False,
) -> dict[str, Any]:
    """Seeded attention fixture with optional causal/padding/all-masked masks."""
    rng = np.random.default_rng(seed)
    dv = value_dim or dim
    scale = 30.0 if extreme_logits else 1.0
    q = rng.standard_normal((sequence, dim)) * scale
    k = rng.standard_normal((sequence, dim)) * scale
    v = rng.standard_normal((sequence, dv))
    mask = None
    if mask_kind == "causal":
        mask = causal_mask(sequence, sequence)
    elif mask_kind == "padding":
        lengths = rng.integers(max(1, sequence // 2), sequence + 1, size=sequence)
        mask = padding_mask(lengths, sequence)
    elif mask_kind == "all_masked":
        mask = np.zeros((sequence, sequence), dtype=bool)
    elif mask_kind == "extreme":
        mask = None
    return {
        "q": q.astype(np.float64),
        "k": k.astype(np.float64),
        "v": v.astype(np.float64),
        "mask": mask,
        "flops": 2 * sequence * sequence * dim + 2 * sequence * sequence * dv,
        "bytes_moved": (2 * sequence * dim + sequence * dv + sequence * dv) * 8
        + (sequence * sequence if mask is not None else 0),
        "description": f"attention fixture seq={sequence} d={dim} mask={mask_kind}",
    }


def causal_mask(rows: int, columns: int) -> np.ndarray:
    """Boolean mask where True keeps the position (query may attend to past keys)."""
    return np.tril(np.ones((rows, columns), dtype=bool))


def padding_mask(lengths: Sequence[int], columns: int) -> np.ndarray:
    """Boolean mask where row i attends to the first lengths[i] keys."""
    mask = np.zeros((len(lengths), columns), dtype=bool)
    for row, length in enumerate(lengths):
        mask[row, : int(length)] = True
    return mask


def low_rank_matrix(
    rows: int = 64,
    columns: int = 48,
    rank: int = 4,
    noise: float = 1e-3,
    seed: int = 21,
) -> dict[str, Any]:
    """Seeded approximately-low-rank matrix fixture."""
    rng = np.random.default_rng(seed)
    left = rng.standard_normal((rows, rank))
    right = rng.standard_normal((rank, columns))
    matrix = left @ right + noise * rng.standard_normal((rows, columns))
    return {
        "A": matrix,
        "true_rank": rank,
        "flops": 2 * rows * columns * min(rows, columns),
        "bytes_moved": rows * columns * 8,
        "description": f"low-rank fixture {rows}x{columns} rank~{rank}",
    }


def least_squares_fixture(
    rows: int = 80,
    columns: int = 12,
    condition: float = 1e3,
    seed: int = 21,
) -> dict[str, Any]:
    """Seeded least-squares fixture with a controlled condition number."""
    rng = np.random.default_rng(seed)
    u, _ = np.linalg.qr(rng.standard_normal((rows, columns)))
    v, _ = np.linalg.qr(rng.standard_normal((columns, columns)))
    singular = np.logspace(0, -np.log10(condition), columns)
    A = (u * singular) @ v.T
    x_true = rng.standard_normal(columns)
    b = A @ x_true + 0.01 * rng.standard_normal(rows)
    return {
        "A": A,
        "b": b,
        "x_true": x_true,
        "condition": float(condition),
        "flops": 2 * rows * columns * columns,
        "bytes_moved": (rows * columns + rows) * 8,
        "description": f"least-squares fixture {rows}x{columns} cond~{condition:.0e}",
    }


def structured_transform_fixture(size: int = 64, seed: int = 21) -> dict[str, Any]:
    """Seeded butterfly-structured operator fixture (size must be a power of two)."""
    from .structured import ButterflyFactors

    if size & (size - 1) or size < 2:
        raise ValueError("butterfly fixture requires a power-of-two size >= 2")
    factors = ButterflyFactors.build(size, seed=seed)
    vector = np.random.default_rng(seed + 1).standard_normal(size)
    return {
        "factors": factors,
        "dense": factors.to_dense(),
        "x": vector,
        "flops": 2 * size * int(np.log2(size)),
        "bytes_moved": size * 8,
        "description": f"butterfly transform fixture size={size}",
    }


def qubo_assignment_fixture(
    tasks: int = 4, resources: int = 5, seed: int = 21, cost_scale: float = 10.0
) -> dict[str, Any]:
    """Seeded assignment problem fixture for classical optimization baselines."""
    rng = np.random.default_rng(seed)
    cost = np.round(rng.uniform(1.0, cost_scale, size=(tasks, resources)), 3)
    return {
        "cost": cost,
        "tasks": tasks,
        "resources": resources,
        "description": f"assignment fixture tasks={tasks} resources={resources}",
    }


FIXTURE_FACTORIES: Mapping[str, Callable[..., dict[str, Any]]] = {
    "attention": attention_fixture,
    "low_rank": low_rank_matrix,
    "least_squares": least_squares_fixture,
    "structured": structured_transform_fixture,
    "qubo": qubo_assignment_fixture,
}


def estimate_optimization_value(
    call_frequency: float,
    per_call_savings_seconds: float,
    setup_cost_seconds: float,
    maintenance_cost_seconds: float = 0.0,
    *,
    complexity_penalty_seconds: float = 0.0,
) -> ValueEstimate:
    """Estimate end-to-end optimization value (integrated step 8)."""
    gross = call_frequency * max(0.0, per_call_savings_seconds)
    net = gross - setup_cost_seconds - maintenance_cost_seconds - complexity_penalty_seconds
    recommend = net > 0 and per_call_savings_seconds > 0
    rationale = (
        f"gross saving {gross:.6f}s over {call_frequency:.3g} calls minus setup "
        f"{setup_cost_seconds:.6f}s, maintenance {maintenance_cost_seconds:.6f}s and "
        f"complexity {complexity_penalty_seconds:.6f}s"
    )
    return ValueEstimate(
        call_frequency=call_frequency,
        per_call_savings_seconds=per_call_savings_seconds,
        setup_cost_seconds=setup_cost_seconds,
        maintenance_cost_seconds=maintenance_cost_seconds,
        net_value_seconds=net,
        recommend_optimization=recommend,
        rationale=rationale,
    )
