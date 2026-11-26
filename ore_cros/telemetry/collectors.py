"""Platform-specific telemetry adapters (ORE Point 23, Steps 1 and 4).

Collectors read real host data where the platform allows it (Linux
``/proc``, cgroup v2, ``os`` process counters) and fall back to a
deterministic simulated source elsewhere. A failing collector is isolated
and reported as missing; it never fabricates zero-valued measurements.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
import platform
import random
import re
import shutil
import time as _time
from typing import Any, Callable, Mapping, Sequence

from ..resource_model import (
    EnvironmentProfile, ResourceKind, ResourceScope, ScopeLevel, machine_scope,
)
from .schemas import Measurement, MetricKind, canonical_unit


def _read_text(path: str | Path) -> str | None:
    """Return file text or None when the path is unreadable or absent."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            return handle.read()
    except (OSError, ValueError):
        return None


def _parse_proc_meminfo() -> dict[str, float] | None:
    """Parse /proc/meminfo into a kibibyte mapping, or None when unavailable."""
    text = _read_text("/proc/meminfo")
    if text is None:
        return None
    values: dict[str, float] = {}
    for line in text.splitlines():
        parts = line.split(":", 1)
        if len(parts) != 2:
            continue
        key = parts[0].strip()
        tokens = parts[1].split()
        if not tokens:
            continue
        try:
            values[key] = float(tokens[0])
        except ValueError:
            continue
    return values or None


def _kernel_at_least(release: str, major: int, minor: int) -> bool:
    """Return True when a kernel release string is at least major.minor."""
    match = re.match(r"(\d+)\.(\d+)", release)
    if match is None:
        return False
    found_major, found_minor = int(match.group(1)), int(match.group(2))
    return (found_major, found_minor) >= (major, minor)


class HostProfiler:
    """Discover the execution environment (spec Step 1).

    Unavailable capacity dimensions are represented as unknown (``None``),
    never as zero, so downstream policy cannot mistake an unmeasured
    resource for an empty one.
    """

    def __init__(self, root: str | Path = "/") -> None:
        """Configure the filesystem root used for discovery (test isolation)."""
        self.root = Path(root)

    def _path(self, relative: str) -> Path:
        """Resolve a path inside the discovery root."""
        return self.root / relative.lstrip("/")

    def discover(
        self, *, profile_version: str = "host-profile-1", queue_capacity: float = 1024.0,
    ) -> EnvironmentProfile:
        """Build the versioned hardware and OS profile for this host."""
        cpu_cores: float | None = None
        try:
            count = os.cpu_count()
            cpu_cores = None if count is None else float(count)
        except (OSError, NotImplementedError):
            cpu_cores = None

        meminfo = _parse_proc_meminfo()
        ram_mib: float | None = None
        if meminfo and "MemTotal" in meminfo:
            ram_mib = meminfo["MemTotal"] / 1024.0

        storage_mib: float | None = None
        try:
            usage = shutil.disk_usage(os.getcwd())
            storage_mib = usage.total / (1024.0 * 1024.0)
        except (OSError, ValueError):
            storage_mib = None

        gpu_available = any(
            self._path(candidate).exists()
            for candidate in (
                "/proc/driver/nvidia/version", "/sys/class/drm", "/dev/nvidia0",
            )
        )

        capacity: dict[ResourceKind, float | None] = {
            ResourceKind.CPU: cpu_cores,
            ResourceKind.RAM: ram_mib,
            ResourceKind.IO: None,
            ResourceKind.NET: None,
            ResourceKind.GPU: None if gpu_available else None,
            ResourceKind.STORAGE: storage_mib,
            ResourceKind.QUEUE: queue_capacity,
        }

        cgroup = self._path("/sys/fs/cgroup/cgroup.controllers").exists()
        kernel_release = self._kernel_release()
        numa_nodes = self._numa_node_count()

        features = {
            "cgroup_v2": cgroup,
            "damon": self._path("/sys/kernel/mm/damon").exists(),
            "io_uring": _kernel_at_least(kernel_release, 5, 1),
            "ebpf": self._path("/sys/fs/bpf").exists(),
            "numa": numa_nodes > 1,
            "accelerator": gpu_available,
        }
        permissions = {
            "proc_stat": self._path("/proc/stat").exists(),
            "proc_meminfo": meminfo is not None,
            "cgroup_cpu_max": self._path("/sys/fs/cgroup/cpu.max").exists(),
            "cgroup_memory": self._path("/sys/fs/cgroup/memory.current").exists(),
        }
        return EnvironmentProfile(
            profile_version=profile_version,
            host_os=platform.system() or "unknown",
            kernel_release=kernel_release,
            capacity=capacity,
            cgroup_support=cgroup,
            features=features,
            telemetry_permissions=permissions,
        )

    def _kernel_release(self) -> str:
        """Return the kernel release string for this host."""
        try:
            return os.uname().release
        except (AttributeError, OSError):
            return platform.release() or "unknown"

    def _numa_node_count(self) -> int:
        """Count NUMA nodes advertised by the platform, or one when absent."""
        base = self._path("/sys/devices/system/node")
        if not base.is_dir():
            return 1
        count = 0
        for entry in base.iterdir():
            if entry.name.startswith("node") and entry.is_dir():
                count += 1
        return max(1, count)


class _BaseCollector:
    """Shared plumbing for collectors that emit machine-scope measurements."""

    def __init__(self, scope: ResourceScope | None = None) -> None:
        """Bind the collector to a scope, defaulting to the local machine."""
        self.scope = scope or machine_scope()

    def collect(self, now: float | None = None) -> list[Measurement]:  # pragma: no cover
        """Return the measurements currently visible to this collector."""
        raise NotImplementedError

    def _measurement(
        self, resource: ResourceKind, metric: MetricKind, value: float,
        source: str, *, now: float, quality: float = 1.0, window: float = 1.0,
    ) -> Measurement:
        """Build one validated measurement, raising on schema violations."""
        return Measurement(
            resource=resource, metric=metric, value=value,
            unit=canonical_unit(resource, metric), timestamp=now, scope=self.scope,
            source=source, quality=quality, window_seconds=window,
        )


class LinuxProcCollector(_BaseCollector):
    """Read /proc counters for machine-scope CPU and memory telemetry.

    CPU utilization is computed from consecutive ``/proc/stat`` samples; the
    first ``collect`` call has no baseline and therefore reports memory only.
    Read errors surface as absent measurements, not as zeros.
    """

    source_name = "/proc"

    def __init__(self, scope: ResourceScope | None = None) -> None:
        """Initialize delta baselines for the CPU utilization series."""
        super().__init__(scope)
        self._previous_cpu_totals: tuple[float, float] | None = None

    def _cpu_sample(self) -> tuple[float, float] | None:
        """Return (busy_ticks, total_ticks) from the aggregate cpu line."""
        text = _read_text("/proc/stat")
        if text is None:
            return None
        for line in text.splitlines():
            if line.startswith("cpu "):
                fields = line.split()[1:]
                try:
                    values = [float(item) for item in fields[:8]]
                except ValueError:
                    return None
                if len(values) < 4:
                    return None
                idle = values[3] + (values[4] if len(values) > 4 else 0.0)
                total = sum(values)
                return total - idle, total
        return None

    def collect(self, now: float | None = None) -> list[Measurement]:
        """Return current machine-scope CPU, memory and pressure measurements."""
        moment = _time.time() if now is None else float(now)
        measurements: list[Measurement] = []
        sample = self._cpu_sample()
        if sample is not None and self._previous_cpu_totals is not None:
            busy_delta = sample[0] - self._previous_cpu_totals[0]
            total_delta = sample[1] - self._previous_cpu_totals[1]
            if total_delta > 0.0 and 0.0 <= busy_delta <= total_delta:
                measurements.append(self._measurement(
                    ResourceKind.CPU, MetricKind.UTILIZATION,
                    busy_delta / total_delta, self.source_name, now=moment,
                ))
        if sample is not None:
            self._previous_cpu_totals = sample

        meminfo = _parse_proc_meminfo()
        if meminfo and "MemTotal" in meminfo and meminfo["MemTotal"] > 0:
            total_mib = meminfo["MemTotal"] / 1024.0
            available = meminfo.get("MemAvailable", 0.0) / 1024.0
            used = max(0.0, total_mib - available)
            measurements.append(self._measurement(
                ResourceKind.RAM, MetricKind.ALLOCATED, used, self.source_name, now=moment,
            ))
            measurements.append(self._measurement(
                ResourceKind.RAM, MetricKind.CAPACITY, total_mib, self.source_name, now=moment,
            ))
            load_text = _read_text("/proc/loadavg")
            if load_text:
                first = load_text.split()[0]
                try:
                    load = float(first)
                    cores = os.cpu_count() or 1
                    pressure = min(1.0, load / max(1, cores))
                    measurements.append(self._measurement(
                        ResourceKind.CPU, MetricKind.PRESSURE, pressure,
                        self.source_name, now=moment, quality=0.9,
                    ))
                except ValueError:
                    pass
        return measurements


class CgroupV2Collector(_BaseCollector):
    """Read cgroup v2 limits and usage for process-scope CPU and memory.

    Quota strings such as ``max 100000`` mean unlimited and are reported as
    unknown capacity (None-valued CAPACITY measurements are omitted).
    """

    source_name = "cgroup.v2"

    def __init__(
        self, cgroup_root: str | Path = "/sys/fs/cgroup",
        scope: ResourceScope | None = None,
    ) -> None:
        """Configure the cgroup mount point used for reads."""
        super().__init__(scope or ResourceScope(ScopeLevel.PROCESS, "self", machine_scope()))
        self.cgroup_root = Path(cgroup_root)
        self._previous_usage_usec: float | None = None

    def _cpu_quota_cores(self) -> float | None:
        """Return the effective CPU core quota, or None when unlimited/absent."""
        text = _read_text(self.cgroup_root / "cpu.max")
        if text is None:
            return None
        parts = text.split()
        if len(parts) != 2 or parts[0] == "max":
            return None
        try:
            quota, period = float(parts[0]), float(parts[1])
        except ValueError:
            return None
        if quota <= 0.0 or period <= 0.0:
            return None
        return quota / period

    def collect(self, now: float | None = None) -> list[Measurement]:
        """Return process-scope cgroup CPU and memory measurements."""
        moment = _time.time() if now is None else float(now)
        measurements: list[Measurement] = []
        quota = self._cpu_quota_cores()
        if quota is not None:
            measurements.append(self._measurement(
                ResourceKind.CPU, MetricKind.CAPACITY, quota, self.source_name, now=moment,
            ))
        usage_text = _read_text(self.cgroup_root / "cpu.stat")
        if usage_text is not None:
            for line in usage_text.splitlines():
                if line.startswith("usage_usec"):
                    try:
                        usage_usec = float(line.split()[1])
                    except (IndexError, ValueError):
                        break
                    if self._previous_usage_usec is not None:
                        measurements.append(self._measurement(
                            ResourceKind.CPU, MetricKind.ALLOCATED,
                            (usage_usec - self._previous_usage_usec) / 1_000_000.0,
                            self.source_name, now=moment,
                        ))
                    self._previous_usage_usec = usage_usec
                    break
        current_text = _read_text(self.cgroup_root / "memory.current")
        max_text = _read_text(self.cgroup_root / "memory.max")
        if current_text is not None:
            try:
                current_mib = float(current_text.strip()) / (1024.0 * 1024.0)
                measurements.append(self._measurement(
                    ResourceKind.RAM, MetricKind.ALLOCATED, current_mib,
                    self.source_name, now=moment,
                ))
            except ValueError:
                pass
        if max_text is not None and max_text.strip() != "max":
            try:
                limit_mib = float(max_text.strip()) / (1024.0 * 1024.0)
                measurements.append(self._measurement(
                    ResourceKind.RAM, MetricKind.CAPACITY, limit_mib,
                    self.source_name, now=moment,
                ))
            except ValueError:
                pass
        return measurements


class ProcessTimesCollector(_BaseCollector):
    """Portable process-scope CPU accounting via ``os.times``."""

    source_name = "os.times"

    def __init__(self, scope: ResourceScope | None = None) -> None:
        """Initialize the delta baseline for CPU time."""
        super().__init__(scope)
        self._previous: tuple[float, float] | None = None

    def collect(self, now: float | None = None) -> list[Measurement]:
        """Return process-scope CPU seconds consumed since the previous call."""
        moment = _time.time() if now is None else float(now)
        times = os.times()
        cpu_seconds = times.user + times.system
        measurements: list[Measurement] = []
        if self._previous is not None:
            delta = max(0.0, cpu_seconds - self._previous)
            measurements.append(self._measurement(
                ResourceKind.CPU, MetricKind.ALLOCATED, delta,
                self.source_name, now=moment, quality=0.85,
            ))
        self._previous = cpu_seconds
        return measurements


@dataclass(frozen=True)
class CollectorStatus:
    """Outcome of one collection attempt from one collector."""

    name: str
    ok: bool
    measurement_count: int
    error: str | None = None


class SimulatedCollector(_BaseCollector):
    """Deterministic simulated telemetry for tests, replay and non-Linux hosts.

    The generator is driven by a seeded PRNG and an optional scripted trace;
    identical seeds and traces produce identical measurement streams.
    """

    source_name = "simulated"

    def __init__(
        self,
        *,
        scope: ResourceScope | None = None,
        seed: int = 7,
        load_profile: Callable[[int], float] | None = None,
        trace: Sequence[Mapping[str, float]] | None = None,
    ) -> None:
        """Configure the synthetic workload generator."""
        super().__init__(scope)
        self._rng = random.Random(seed)
        self._step = 0
        self._load_profile = load_profile or (lambda step: 0.5)
        self._trace = [dict(item) for item in trace] if trace is not None else None

    def _load(self) -> dict[str, float]:
        """Return the synthetic load vector for the current step."""
        if self._trace is not None:
            index = min(self._step, len(self._trace) - 1)
            row = self._trace[index] if self._trace else {}
            return {str(key): float(value) for key, value in row.items()}
        level = min(1.5, max(0.0, self._load_profile(self._step)))
        jitter = self._rng.uniform(-0.02, 0.02)
        return {
            "cpu": min(1.5, max(0.0, level + jitter)),
            "ram": min(1.0, max(0.0, level * 0.8 + jitter)),
            "io": min(1.0, max(0.0, level * 0.6 + jitter)),
            "net": min(1.0, max(0.0, level * 0.5 + jitter)),
            "pressure": min(1.0, max(0.0, level - 0.5 + jitter)),
            "queue_depth": float(int(max(0.0, (level - 0.4) * 20.0))),
            "arrival_rate": max(0.0, level * 2.0),
            "wait_seconds": max(0.0, (level - 0.3) * 2.0),
        }

    def collect(self, now: float | None = None) -> list[Measurement]:
        """Return one deterministic synthetic measurement batch."""
        moment = _time.time() if now is None else float(now)
        load = self._load()
        self._step += 1
        measurements = [
            self._measurement(ResourceKind.CPU, MetricKind.UTILIZATION, load["cpu"],
                              self.source_name, now=moment, quality=0.75),
            self._measurement(ResourceKind.RAM, MetricKind.UTILIZATION, load["ram"],
                              self.source_name, now=moment, quality=0.75),
            self._measurement(ResourceKind.IO, MetricKind.UTILIZATION, load["io"],
                              self.source_name, now=moment, quality=0.7),
            self._measurement(ResourceKind.NET, MetricKind.UTILIZATION, load["net"],
                              self.source_name, now=moment, quality=0.7),
            self._measurement(ResourceKind.CPU, MetricKind.PRESSURE, load["pressure"],
                              self.source_name, now=moment, quality=0.7),
            self._measurement(ResourceKind.QUEUE, MetricKind.QUEUE_DEPTH,
                              load["queue_depth"], self.source_name, now=moment, quality=0.7),
            self._measurement(ResourceKind.QUEUE, MetricKind.ARRIVAL_RATE,
                              load["arrival_rate"], self.source_name, now=moment, quality=0.7),
            self._measurement(ResourceKind.QUEUE, MetricKind.WAIT_SECONDS,
                              load["wait_seconds"], self.source_name, now=moment, quality=0.7),
        ]
        return measurements


class CollectorSet:
    """Compose collectors with per-collector isolation and deduplication."""

    def __init__(self, collectors: Sequence[Any]) -> None:
        """Store collectors under their declared names."""
        if not collectors:
            raise ValueError("at least one collector is required")
        self.collectors: dict[str, Any] = {}
        for collector in collectors:
            name = getattr(collector, "source_name", type(collector).__name__)
            if name in self.collectors:
                raise ValueError(f"duplicate collector name {name!r}")
            self.collectors[name] = collector

    def collect(self, now: float | None = None) -> tuple[list[Measurement], list[CollectorStatus]]:
        """Collect from every collector, isolating crashes per collector.

        Measurements are deduplicated per (scope, resource, metric): the
        highest-quality source wins and ties resolve deterministically by
        source name order.
        """
        moment = _time.time() if now is None else float(now)
        statuses: list[CollectorStatus] = []
        candidates: list[Measurement] = []
        for name in sorted(self.collectors):
            collector = self.collectors[name]
            try:
                produced = collector.collect(now=moment)
            except Exception as exc:  # noqa: BLE001 - collector crash isolation
                statuses.append(CollectorStatus(name, False, 0, f"{type(exc).__name__}: {exc}"))
                continue
            produced = list(produced)
            candidates.extend(produced)
            statuses.append(CollectorStatus(name, True, len(produced)))
        deduplicated = self._deduplicate(candidates)
        return deduplicated, statuses

    @staticmethod
    def _deduplicate(candidates: Sequence[Measurement]) -> list[Measurement]:
        """Keep the highest-quality measurement per (scope, resource, metric)."""
        best: dict[tuple[str, str, str, str], Measurement] = {}
        for measurement in sorted(candidates, key=lambda item: item.source):
            key = (
                measurement.scope.level.value, measurement.scope.scope_id,
                measurement.resource.value, measurement.metric.value,
            )
            current = best.get(key)
            if current is None or measurement.quality > current.quality:
                best[key] = measurement
        return sorted(best.values(), key=lambda item: (
            item.scope.level.value, item.scope.scope_id,
            item.resource.value, item.metric.value,
        ))


@dataclass
class BaselineProfile:
    """Baseline measurements for policy comparison (spec Step 4)."""

    phases: dict[str, list[Measurement]] = field(default_factory=dict)
    overhead_seconds: float = 0.0
    missing_rate: float = 0.0

    def phase_summary(self, phase: str) -> dict[str, list[float]]:
        """Return raw values per (resource, metric) for one labelled phase."""
        summary: dict[str, list[float]] = {}
        for measurement in self.phases.get(phase, []):
            key = f"{measurement.resource.value}/{measurement.metric.value}"
            summary.setdefault(key, []).append(measurement.value)
        return summary

    def to_dict(self) -> dict[str, Any]:
        """Return a detached, JSON-compatible snapshot of the baseline."""
        return {
            "phases": {
                phase: [item.to_dict() for item in measurements]
                for phase, measurements in self.phases.items()
            },
            "overhead_seconds": self.overhead_seconds,
            "missing_rate": self.missing_rate,
        }


def record_baseline(
    collector_set: CollectorSet,
    *,
    phases: Sequence[str] = ("idle", "startup", "steady", "burst"),
    samples_per_phase: int = 3,
    now: float | None = None,
) -> BaselineProfile:
    """Sample the collector set under caller-labelled workload phases.

    Measurement overhead and missing-source rates are recorded so telemetry
    cost itself stays observable (spec Step 4). Phases are labels supplied
    by the caller; this function does not synthesize workload levels.
    """
    if samples_per_phase <= 0:
        raise ValueError("samples_per_phase must be positive")
    profile = BaselineProfile()
    total_measurements = 0
    total_failures = 0
    started = _time.perf_counter()
    moment = _time.time() if now is None else float(now)
    for phase in phases:
        for _ in range(samples_per_phase):
            measurements, statuses = collector_set.collect(now=moment)
            profile.phases.setdefault(phase, []).extend(measurements)
            total_measurements += sum(status.measurement_count for status in statuses)
            total_failures += sum(0 if status.ok else 1 for status in statuses)
    profile.overhead_seconds = _time.perf_counter() - started
    attempts = max(1, len(phases) * samples_per_phase * len(collector_set.collectors))
    profile.missing_rate = total_failures / attempts
    return profile


def default_machine_collectors(
    profile: EnvironmentProfile, *,
    include_simulated: bool = True,
) -> CollectorSet:
    """Build the default machine-scope collector set for this platform."""
    collectors: list[Any] = []
    if profile.host_os.lower() == "linux":
        collectors.append(LinuxProcCollector())
        if profile.cgroup_support:
            collectors.append(CgroupV2Collector())
    collectors.append(
        ProcessTimesCollector(ResourceScope(ScopeLevel.PROCESS, "self", machine_scope()))
    )
    if include_simulated or not collectors:
        collectors.append(SimulatedCollector())
    return CollectorSet(collectors)


__all__ = [
    "HostProfiler", "LinuxProcCollector", "CgroupV2Collector", "ProcessTimesCollector",
    "SimulatedCollector", "CollectorSet", "CollectorStatus", "BaselineProfile",
    "record_baseline", "default_machine_collectors", "canonical_unit",
]
