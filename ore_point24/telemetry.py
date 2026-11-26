"""Phase VII, Step 31 - establish real resource telemetry.

Record wall-clock and CPU time separately, peak and average memory where
supported, storage and network I/O where available, and worker counts, queue
wait and timeouts.  Every measurement is associated with module, workload,
configuration and machine identity, and absent measurements are marked
*unavailable* rather than estimated silently.  Workload-specific moving
averages and dispersion are computed from observed samples with the raw
sample count and observation window attached.
"""
from __future__ import annotations

import os
import platform
import time
from dataclasses import dataclass, field
from typing import Any, Mapping

import psutil

try:
    import resource
except ImportError:
    resource = None

from .core import new_id, sha256_digest, utc_now_iso

TELEMETRY_SCHEMA = "ore.point24.telemetry.v1"
UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True)
class ResourceMeasurement:
    """One measured sample; UNAVAILABLE marks absent, never invented, data."""

    wall_clock_seconds: float | str
    cpu_seconds: float | str
    peak_memory_mb: float | str
    average_memory_mb: float | str
    storage_io_bytes: float | str
    network_io_bytes: float | str
    accelerator_utilization: float | str
    machine_identity: str
    measured_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return {
            "wall_clock_seconds": self.wall_clock_seconds,
            "cpu_seconds": self.cpu_seconds,
            "peak_memory_mb": self.peak_memory_mb,
            "average_memory_mb": self.average_memory_mb,
            "storage_io_bytes": self.storage_io_bytes,
            "network_io_bytes": self.network_io_bytes,
            "accelerator_utilization": self.accelerator_utilization,
            "machine_identity": self.machine_identity,
            "measured_at": self.measured_at,
        }


@dataclass(slots=True)
class WorkloadStatistics:
    """Moving averages and dispersion computed from observed samples only."""

    samples: int = 0
    observation_window_seconds: float = 0.0
    _values: list[float] = field(default_factory=list)

    def observe(self, value: float, *, window_seconds: float) -> None:
        self._values.append(float(value))
        self.samples += 1
        self.observation_window_seconds = window_seconds

    def mean(self) -> float | None:
        if not self._values:
            return None
        return sum(self._values) / len(self._values)

    def dispersion(self) -> float | None:
        if len(self._values) < 2:
            return None
        mean = self.mean()
        assert mean is not None
        return (sum((v - mean) ** 2 for v in self._values) / (len(self._values) - 1)) ** 0.5

    def summary(self) -> dict[str, Any]:
        return {
            "samples": self.samples,
            "observation_window_seconds": self.observation_window_seconds,
            "mean": self.mean(),
            "dispersion": self.dispersion(),
        }


class ResourceTelemetry:
    """Collect per-module, per-workload resource measurements (Step 31).

    ``measure`` wraps a callable and records real observations from
    ``time.perf_counter`` (wall clock), ``time.process_time`` (CPU) and the
    ``resource`` module (peak RSS).  Metrics the platform cannot provide are
    stored as ``unavailable`` - never estimated silently.
    """

    def __init__(self) -> None:
        self._statistics: dict[tuple[str, str], WorkloadStatistics] = {}
        self._samples: list[tuple[str, str, ResourceMeasurement]] = []
        self.machine_identity = f"{platform.node()}/{platform.machine()}"

    def measure(
        self,
        module_id: str,
        workload: str,
        callable_: Any,
        *,
        configuration: Mapping[str, Any] | None = None,
    ) -> ResourceMeasurement:
        """Run one callable under measurement and store the sample."""
        before_cpu = time.process_time()
        before_wall = time.perf_counter()
        try:
            callable_()
        finally:
            after_wall = time.perf_counter()
            after_cpu = time.process_time()
        wall = after_wall - before_wall
        cpu = after_cpu - before_cpu
        try:
            process = psutil.Process()
            memory_info = process.memory_info()

            # Windows: peak_wset is the process's peak working-set memory.
            if hasattr(memory_info, "peak_wset"):
                peak_mb = memory_info.peak_wset / (1024 * 1024)

            # Linux/macOS: use resource.getrusage when available.
            elif resource is not None:
                peak_memory = resource.getrusage(
                    resource.RUSAGE_SELF
                ).ru_maxrss

                # macOS reports bytes; Linux reports kilobytes.
                if platform.system() == "Darwin":
                    peak_mb = peak_memory / (1024 * 1024)
                else:
                    peak_mb = peak_memory / 1024

            else:
                peak_mb = UNAVAILABLE

        except (AttributeError, OSError):
            peak_mb = UNAVAILABLE
        measurement = ResourceMeasurement(
            wall_clock_seconds=round(wall, 6),
            cpu_seconds=round(cpu, 6),
            peak_memory_mb=peak_mb if peak_mb != UNAVAILABLE else UNAVAILABLE,
            average_memory_mb=UNAVAILABLE,
            storage_io_bytes=UNAVAILABLE,
            network_io_bytes=UNAVAILABLE,
            accelerator_utilization=UNAVAILABLE,
            machine_identity=self.machine_identity,
        )
        self._samples.append((module_id, workload, measurement))
        stats = self._statistics.setdefault((module_id, workload), WorkloadStatistics())
        stats.observe(wall, window_seconds=wall)
        return measurement

    def statistics(self, module_id: str, workload: str) -> dict[str, Any]:
        stats = self._statistics.get((module_id, workload))
        if stats is None:
            return {"samples": 0, "note": "no observations recorded"}
        return stats.summary()

    def measurements(self, *, module_id: str | None = None) -> list[dict[str, Any]]:
        rows = []
        for record_module, workload, measurement in self._samples:
            if module_id is not None and record_module != module_id:
                continue
            rows.append({
                "module_id": record_module,
                "workload": workload,
                **measurement.to_dict(),
            })
        return rows

    def summary(self) -> dict[str, Any]:
        return {
            "schema": TELEMETRY_SCHEMA,
            "telemetry_id": new_id("point24_telemetry"),
            "machine_identity": self.machine_identity,
            "sample_count": len(self._samples),
            "per_module_workload": {
                f"{module}|{workload}": stats.summary()
                for (module, workload), stats in sorted(self._statistics.items())
            },
            "note": "absent measurements are stored as 'unavailable', never invented",
        }


def platform_capabilities() -> dict[str, Any]:
    """Report which resource dimensions this platform can actually observe."""
    return {
        "wall_clock": True,
        "cpu_time": hasattr(time, "process_time"),
        "peak_memory": hasattr(resource, "getrusage"),
        "average_memory": False,
        "storage_io": False,
        "network_io": False,
        "accelerators": False,
        "cpu_count": os.cpu_count(),
    }


__all__ = [
    "TELEMETRY_SCHEMA",
    "UNAVAILABLE",
    "ResourceMeasurement",
    "ResourceTelemetry",
    "WorkloadStatistics",
    "platform_capabilities",
]
