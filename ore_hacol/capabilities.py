"""Hardware capability discovery and calibration (integrated step 13).

Discovers CPU vector features, core counts, memory capacity/availability, a
measured memory-bandwidth estimate, a measured dense-GEMM compute-rate
estimate, accelerator availability, supported dtypes, runtime versions and
current utilization. Numbers are measured where possible and labelled as
estimates otherwise; nothing here claims a GPU or quantum device is present.
"""
from __future__ import annotations

import os
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np

from .contracts import HardwareProfile

_CPU_INFO = Path("/proc/cpuinfo")
_MEM_INFO = Path("/proc/meminfo")
_LOADAVG = Path("/proc/loadavg")


def _read_cpu_features() -> tuple[tuple[str, ...], str]:
    """Parse CPU model name and instruction-set flags from /proc/cpuinfo."""
    model = "unknown"
    features: set[str] = set()
    try:
        text = _CPU_INFO.read_text()
        for line in text.splitlines():
            if line.startswith("model name") and model == "unknown":
                model = line.split(":", 1)[1].strip()
            elif line.startswith("flags") or line.startswith("Features"):
                features.update(line.split(":", 1)[1].split())
    except OSError:
        pass
    if model == "unknown":
        model = f"{platform.machine()} cpu"
    return tuple(sorted(features)), model


def _read_memory_bytes() -> tuple[int, int]:
    """Return (capacity, available) physical memory in bytes from /proc/meminfo."""
    capacity = available = 0
    try:
        text = _MEM_INFO.read_text()
        for line in text.splitlines():
            if line.startswith("MemTotal:"):
                capacity = int(line.split()[1]) * 1024
            elif line.startswith("MemAvailable:"):
                available = int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return capacity, available


def _read_load_average() -> float:
    """Return the 1-minute load average when /proc/loadavg is readable."""
    try:
        return float(_LOADAVG.read_text().split()[0])
    except (OSError, ValueError, IndexError):
        return -1.0


def measure_bandwidth_bytes_per_second(size_bytes: int = 1 << 24, repeats: int = 3) -> float:
    """Measure sustained memory bandwidth by copying one large buffer repeatedly.

    This is a measured estimate of streaming bandwidth for the current process,
    not a vendor specification. Bytes counted are read + write per copy.
    """
    if size_bytes < 1 << 16:
        size_bytes = 1 << 16
    elements = max(1, size_bytes // 8)
    source = np.ones(elements, dtype=np.float64)
    destination = np.empty_like(source)
    best = 0.0
    for _ in range(max(1, repeats)):
        start = time.perf_counter()
        destination[:] = source
        elapsed = max(time.perf_counter() - start, 1e-9)
        moved = 2 * elements * 8
        best = max(best, moved / elapsed)
    return float(best)


def measure_compute_rate_flops_per_second(size: int = 256, repeats: int = 3) -> float:
    """Measure a dense double-precision GEMM rate as the compute-rate estimate.

    Uses small square matrices so calibration stays fast; the number is an
    engineering estimate of available throughput for the current process.
    """
    a = np.asarray(np.random.default_rng(0).standard_normal((size, size)), dtype=np.float64)
    b = np.asarray(np.random.default_rng(1).standard_normal((size, size)), dtype=np.float64)
    best = 0.0
    # Warm up caches and allocator before timing.
    for _ in range(1):
        a @ b
    for _ in range(max(1, repeats)):
        start = time.perf_counter()
        a @ b
        elapsed = max(time.perf_counter() - start, 1e-9)
        flops = 2.0 * size**3
        best = max(best, flops / elapsed)
    return float(best)


def _detect_accelerators() -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Detect optional accelerators without importing heavyweight libraries.

    HACOL treats accelerators as optional backends; absent devices are reported
    honestly instead of being assumed present.
    """
    kinds: list[str] = []
    notes: list[str] = []
    if Path("/dev/nvidia0").exists():  # pragma: no cover - depends on host
        kinds.append("cuda")
    if os.environ.get("ORE_HACOL_ACCELERATORS"):  # pragma: no cover - opt-in
        kinds.extend(
            item.strip() for item in os.environ["ORE_HACOL_ACCELERATORS"].split(",") if item.strip()
        )
    if not kinds:
        notes.append("No accelerator detected; CPU reference and optimized CPU backends only.")
    return tuple(kinds), tuple(notes)


def discover_hardware_profile(
    *,
    calibrate: bool = True,
    bandwidth_bytes: int = 1 << 22,
    compute_size: int = 192,
) -> HardwareProfile:
    """Discover the current hardware capabilities (integrated step 13).

    Calibration sizes default to small, bounded workloads so discovery costs
    well under a second in ordinary use; pass larger sizes for more accurate
    rates when driving long optimization campaigns.
    """
    features, model = _read_cpu_features()
    capacity, available = _read_memory_bytes()
    accelerators, accel_notes = _detect_accelerators()
    import numpy

    runtime_versions = {
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "machine": platform.machine(),
        "system": platform.system(),
    }
    bandwidth = measure_bandwidth_bytes_per_second(bandwidth_bytes) if calibrate else 0.0
    compute_rate = measure_compute_rate_flops_per_second(compute_size) if calibrate else 0.0
    load = _read_load_average()
    utilization = {"load_average_1min": load} if load >= 0 else {}
    notes = list(accel_notes)
    notes.append(
        "bandwidth and compute rates are measured microbenchmark estimates for the "
        "current process, not vendor specifications"
    )
    return HardwareProfile(
        cpu_model=model,
        cpu_features=features,
        core_count=os.cpu_count() or 1,
        memory_capacity_bytes=capacity,
        memory_available_bytes=available,
        bandwidth_bytes_per_second=bandwidth,
        compute_rate_flops_per_second=compute_rate,
        accelerator_types=accelerators,
        supported_dtypes=("float16", "bfloat16", "float32", "float64"),
        runtime_versions=runtime_versions,
        utilization=utilization,
        discovery_notes=tuple(notes),
    )


def calibrate_profile(profile: HardwareProfile) -> HardwareProfile:
    """Return a profile with measured bandwidth/compute rates when missing."""
    if profile.bandwidth_bytes_per_second > 0 and profile.compute_rate_flops_per_second > 0:
        return profile
    from dataclasses import replace

    return replace(
        profile,
        bandwidth_bytes_per_second=(
            profile.bandwidth_bytes_per_second or measure_bandwidth_bytes_per_second(1 << 22)
        ),
        compute_rate_flops_per_second=(
            profile.compute_rate_flops_per_second or measure_compute_rate_flops_per_second(192)
        ),
    )


def capability_summary(profile: HardwareProfile) -> dict[str, Any]:
    """Human/Blackboard-friendly capability summary (identity + key rates)."""
    return {
        "hardware_identity": profile.hardware_identity(),
        "cpu_model": profile.cpu_model,
        "core_count": profile.core_count,
        "vector_features_selected": [
            feature
            for feature in ("avx", "avx2", "avx512f", "sse4_2", "neon")
            if feature in profile.cpu_features
        ],
        "memory_capacity_bytes": profile.memory_capacity_bytes,
        "measured_bandwidth_bytes_per_second": round(profile.bandwidth_bytes_per_second, 1),
        "measured_compute_rate_gflops": round(
            profile.compute_rate_flops_per_second / 1e9, 3
        ),
        "accelerator_types": list(profile.accelerator_types),
        "runtime_versions": dict(profile.runtime_versions),
    }
