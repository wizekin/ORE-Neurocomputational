"""Intelligent memory management policies (ORE Point 23, Steps 27-35).

Covers working-set estimation, application-level cache tiers, LRU/CLOCK
baselines, an MGLRU-inspired multi-generation replacer, a read-only DAMON
adapter, NUMA-aware placement, compression/tiering decisions, accelerator
memory admission and a CXL extension point kept deliberately minimal.
"""
from __future__ import annotations

from collections import OrderedDict, defaultdict, deque
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


class WorkingSetEstimator:
    """Estimate per-workload working sets (Step 27).

    Tracks object accesses over short and medium windows; the working-set
    size is the number of distinct objects touched in the window, and the
    rate of change separates sustained demand from temporary peaks.
    """

    def __init__(self, *, short_window: float = 1.0, medium_window: float = 10.0) -> None:
        """Configure the two observation windows."""
        if not 0.0 < short_window < medium_window:
            raise ValueError("windows must satisfy 0 < short_window < medium_window")
        self.short_window = short_window
        self.medium_window = medium_window
        self._accesses: dict[str, deque[tuple[float, str]]] = defaultdict(deque)
        self._last_estimate: dict[str, tuple[float, int]] = {}

    def observe(self, task_id: str, object_id: str, *, now: float) -> None:
        """Record one object access for a task."""
        if not object_id:
            raise ValueError("object_id must be non-empty")
        self._accesses[task_id].append((now, object_id))

    def working_set(self, task_id: str, *, window: float, now: float) -> frozenset[str]:
        """Return the set of objects accessed within the trailing window."""
        history = self._accesses.get(task_id, deque())
        cutoff = now - window
        return frozenset(
            object_id for timestamp, object_id in history if timestamp >= cutoff
        )

    def estimate(self, task_id: str, *, now: float) -> "WorkingSetEstimate":
        """Return the working-set estimate with sustained-vs-peak evidence.

        The rate of change compares the current short-window size against
        the previous estimate for the same task, divided by elapsed time.
        """
        short = self.working_set(task_id, window=self.short_window, now=now)
        medium = self.working_set(task_id, window=self.medium_window, now=now)
        previous = self._last_estimate.get(task_id)
        if previous is not None and now > previous[0]:
            rate_of_change = (len(short) - previous[1]) / (now - previous[0])
        else:
            rate_of_change = 0.0
        self._last_estimate[task_id] = (now, len(short))
        sustained = len(short) > 0 and len(short) >= 0.8 * max(1, len(medium))
        return WorkingSetEstimate(
            task_id=task_id, short_window_size=len(short),
            medium_window_size=len(medium), rate_of_change=rate_of_change,
            sustained=sustained, peak_size=len(medium),
        )

    def prune(self, *, now: float) -> int:
        """Drop accesses older than the medium window; return removed count."""
        removed = 0
        cutoff = now - self.medium_window
        for history in self._accesses.values():
            while history and history[0][0] < cutoff:
                history.popleft()
                removed += 1
        return removed


@dataclass(frozen=True)
class WorkingSetEstimate:
    """Working-set size evidence for one workload (Step 27)."""

    task_id: str
    short_window_size: int
    medium_window_size: int
    rate_of_change: float
    sustained: bool
    peak_size: int

    def to_dict(self) -> dict[str, Any]:
        """Return a detached, JSON-compatible snapshot of the estimate."""
        return {
            "task_id": self.task_id, "short_window_size": self.short_window_size,
            "medium_window_size": self.medium_window_size,
            "rate_of_change": self.rate_of_change, "sustained": self.sustained,
            "peak_size": self.peak_size,
        }


@dataclass
class CachedEntry:
    """One cache entry carrying value and provenance through tier moves."""

    value: Any
    provenance: str
    frequency: int = 0


class TieredCache:
    """Hot/warm/cold application-level cache tiers (Step 28).

    Tier moves never change the retrieved value or its provenance; hotter
    tiers hold frequently and recently used entries, colder tiers hold the
    rest. Eviction only happens from the cold tier.
    """

    _TIER_ORDER = ("hot", "warm", "cold")

    def __init__(self, *, hot: int = 4, warm: int = 8, cold: int = 16) -> None:
        """Configure tier capacities."""
        for name, size in (("hot", hot), ("warm", warm), ("cold", cold)):
            if size < 0:
                raise ValueError(f"{name} capacity must be non-negative")
        self._capacity = {"hot": hot, "warm": warm, "cold": cold}
        self._tiers: dict[str, OrderedDict[str, CachedEntry]] = {
            tier: OrderedDict() for tier in self._TIER_ORDER
        }
        self._tier_moves: list[dict[str, Any]] = []
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Any | None:
        """Return the cached value regardless of tier; promotion recorded.

        Retrieval semantics and provenance are identical across tiers; only
        residency changes. Returns None on a miss.
        """
        for tier, entries in self._tiers.items():
            if key in entries:
                entry = entries.pop(key)
                entry.frequency += 1
                self._hits += 1
                target = self._target_tier(entry.frequency)
                if target != tier:
                    self._record_move(key, tier, target)
                self._tiers[target][key] = entry
                self._enforce_capacity(target)
                return entry.value
        self._misses += 1
        return None

    def provenance_of(self, key: str) -> str | None:
        """Return the provenance string stored with one key, if present."""
        for entries in self._tiers.values():
            if key in entries:
                return entries[key].provenance
        return None

    def put(self, key: str, value: Any, *, provenance: str = "unknown") -> None:
        """Insert or refresh one entry, starting in the cold tier."""
        if not isinstance(key, str) or not key:
            raise ValueError("key must be a non-empty string")
        for entries in self._tiers.values():
            entries.pop(key, None)
        self._tiers["cold"][key] = CachedEntry(value=value, provenance=provenance)
        self._enforce_capacity("cold")

    def _target_tier(self, frequency: int) -> str:
        """Map access frequency to a target tier (promotion policy)."""
        if frequency >= 3:
            return "hot"
        if frequency == 2:
            return "warm"
        return "cold"

    def _record_move(self, key: str, source: str, target: str) -> None:
        """Record one tier transition for auditability."""
        self._tier_moves.append({"key": key, "from": source, "to": target})

    def _enforce_capacity(self, tier: str) -> None:
        """Evict from the cold tier only, when capacity is exceeded."""
        entries = self._tiers[tier]
        capacity = self._capacity[tier]
        while len(entries) > capacity:
            if tier == "cold":
                entries.popitem(last=False)
            else:
                demoted_key, demoted = entries.popitem(last=False)
                lower = self._TIER_ORDER[self._TIER_ORDER.index(tier) + 1]
                self._record_move(demoted_key, tier, lower)
                self._tiers[lower][demoted_key] = demoted
                self._enforce_capacity(lower)
                break

    def tier_sizes(self) -> dict[str, int]:
        """Return the number of entries currently in each tier."""
        return {tier: len(entries) for tier, entries in self._tiers.items()}

    def tier_of(self, key: str) -> str | None:
        """Return the tier currently holding one key."""
        for tier, entries in self._tiers.items():
            if key in entries:
                return tier
        return None

    def hit_rate(self) -> float:
        """Return the overall hit rate across all tiers."""
        total = self._hits + self._misses
        return self._hits / total if total else 0.0

    def tier_moves(self) -> tuple[dict[str, Any], ...]:
        """Return the recorded tier transitions."""
        return tuple(dict(item) for item in self._tier_moves)


class LRUReplacer:
    """Plain least-recently-used replacement baseline (Step 29).

    Not scan resistant: one large sequential scan evicts the entire
    working set; ``hit_rate`` makes that measurable for comparison.
    """

    def __init__(self, capacity: int) -> None:
        """Configure the number of slots."""
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._entries: OrderedDict[str, None] = OrderedDict()
        self._hits = 0
        self._misses = 0

    def touch(self, key: str, *, inserted: bool = False) -> bool:
        """Record one access; return True when the key was resident."""
        if key in self._entries:
            self._entries.move_to_end(key)
            self._hits += 1
            return True
        self._misses += 1
        if inserted:
            self._entries[key] = None
            while len(self._entries) > self.capacity:
                self._entries.popitem(last=False)
        return False

    def evict(self) -> str | None:
        """Return and remove the least recently used key."""
        if not self._entries:
            return None
        key, _ = self._entries.popitem(last=False)
        return key

    def hit_rate(self) -> float:
        """Return the hit rate across all recorded accesses."""
        total = self._hits + self._misses
        return self._hits / total if total else 0.0

    @property
    def metadata_overhead_entries(self) -> int:
        """Return the number of bookkeeping entries (metadata overhead)."""
        return len(self._entries)


class CLOCKReplacer:
    """Second-chance CLOCK replacement baseline (Step 29)."""

    def __init__(self, capacity: int) -> None:
        """Configure the number of slots."""
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._frames: list[list[Any]] = []  # [key, reference_bit]
        self._hand = 0
        self._hits = 0
        self._misses = 0

    def touch(self, key: str, *, inserted: bool = False) -> bool:
        """Record one access; set the reference bit on resident frames."""
        for frame in self._frames:
            if frame[0] == key:
                frame[1] = True
                self._hits += 1
                return True
        self._misses += 1
        if inserted:
            if len(self._frames) < self.capacity:
                self._frames.append([key, True])
            else:
                self._sweep()
                self._frames[self._hand] = [key, True]
                self._hand = (self._hand + 1) % self.capacity
        return False

    def _sweep(self) -> None:
        """Advance the hand, clearing reference bits until a victim is found."""
        while True:
            frame = self._frames[self._hand]
            if frame[1]:
                frame[1] = False
                self._hand = (self._hand + 1) % self.capacity
            else:
                return

    def hit_rate(self) -> float:
        """Return the hit rate across all recorded accesses."""
        total = self._hits + self._misses
        return self._hits / total if total else 0.0


class MultiGenerationReplacer:
    """MGLRU-inspired multi-generation replacement (Step 30).

    Younger generations hold more recently used objects; eviction drains
    the oldest generation first. Accesses promote objects into the youngest
    generation, and insertions age all generations at a fixed interval,
    giving scan resistance that plain LRU lacks.
    """

    def __init__(
        self, *, capacity: int, generations: int = 4, aging_interval: int = 4,
    ) -> None:
        """Configure total capacity, generation count and the aging interval."""
        if capacity <= 0 or generations <= 0 or aging_interval <= 0:
            raise ValueError("capacity, generations and aging_interval must be positive")
        self.capacity = capacity
        self.generations = generations
        self.aging_interval = aging_interval
        self._rings: list[set[str]] = [set() for _ in range(generations)]
        self._insert_count = 0
        self._hits = 0
        self._misses = 0

    def touch(self, key: str, *, inserted: bool = False) -> bool:
        """Record one access, promoting the key to the youngest generation."""
        for ring in self._rings:
            if key in ring:
                ring.remove(key)
                self._rings[0].add(key)
                self._hits += 1
                return True
        self._misses += 1
        if inserted:
            self._rings[0].add(key)
            self._insert_count += 1
            if self._insert_count % self.aging_interval == 0:
                self._rotate_generations()
            self._evict_beyond_capacity()
        return False

    def _rotate_generations(self) -> None:
        """Age every generation one level; the oldest generation spills."""
        spilled = self._rings[self.generations - 1]
        for index in range(self.generations - 1, 0, -1):
            self._rings[index] = self._rings[index - 1]
        self._rings[0] = set()
        if spilled:
            self._rings[self.generations - 1].update(spilled)
            self._evict_beyond_capacity()

    def _evict_beyond_capacity(self) -> None:
        """Drop keys from the oldest generations until within capacity."""
        total = sum(len(ring) for ring in self._rings)
        while total > self.capacity:
            for index in range(self.generations - 1, -1, -1):
                if self._rings[index]:
                    self._rings[index].pop()
                    total -= 1
                    break
            else:  # pragma: no cover - capacity guard makes this unreachable
                return

    def evict(self) -> str | None:
        """Return one key from the oldest non-empty generation."""
        for index in range(self.generations - 1, -1, -1):
            if self._rings[index]:
                return self._rings[index].pop()
        return None

    def generation_sizes(self) -> list[int]:
        """Return the current size of every generation, youngest first."""
        return [len(ring) for ring in self._rings]

    def hit_rate(self) -> float:
        """Return the hit rate across all recorded accesses."""
        total = self._hits + self._misses
        return self._hits / total if total else 0.0


class DamonAdapter:
    """Read-only DAMON access-monitoring adapter (Step 31).

    Where the kernel exposes DAMON, the adapter reads existing scheme
    status read-only; it never installs or modifies operation schemes.
    When DAMON is unavailable the adapter reports that explicitly instead
    of fabricating access patterns.
    """

    def __init__(self, damon_root: str | Path = "/sys/kernel/mm/damon") -> None:
        """Configure the DAMON sysfs root."""
        self.damon_root = Path(damon_root)

    def available(self) -> bool:
        """Return True when the DAMON interface exists on this platform."""
        return self.damon_root.exists()

    def read_access_pattern(self) -> "DamonReading":
        """Return a read-only access-pattern status, never mutating state."""
        if not self.available():
            return DamonReading(
                available=False, note="DAMON interface not present on this platform",
            )
        targets = [
            entry.name for entry in self.damon_root.iterdir() if entry.is_dir()
        ] if self.damon_root.is_dir() else []
        return DamonReading(
            available=True, targets=tuple(sorted(targets)),
            note="read-only inspection of existing DAMON interfaces",
        )


@dataclass(frozen=True)
class DamonReading:
    """Outcome of one read-only DAMON inspection (Step 31)."""

    available: bool
    targets: tuple[str, ...] = ()
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return a detached, JSON-compatible snapshot of the reading."""
        return {
            "available": self.available, "targets": list(self.targets), "note": self.note,
        }


@dataclass(frozen=True)
class NumaNode:
    """One NUMA node with its CPU list and memory capacity (Step 32)."""

    node_id: int
    cpus: tuple[int, ...]
    memory_mib: float


class NumaTopology:
    """NUMA topology and locality-aware placement costs (Step 32).

    When the platform exposes multiple nodes, placement cost includes the
    remote-access penalty; on single-node or non-Linux systems locality is
    explicitly reported unavailable rather than assumed.
    """

    REMOTE_PENALTY = 1.7  # remote access costs ~1.7x local on typical servers

    def __init__(self, nodes: Sequence[NumaNode] | None = None) -> None:
        """Register discovered nodes or leave the topology unavailable."""
        self.nodes: tuple[NumaNode, ...] = tuple(nodes or ())
        if len({node.node_id for node in self.nodes}) != len(self.nodes):
            raise ValueError("NUMA node ids must be unique")

    @classmethod
    def discover(cls, root: str | Path = "/sys/devices/system/node") -> "NumaTopology":
        """Discover NUMA nodes from sysfs; empty when unavailable."""
        base = Path(root)
        nodes: list[NumaNode] = []
        if not base.is_dir():
            return cls(nodes)
        for entry in sorted(base.iterdir()):
            if not (entry.name.startswith("node") and entry.is_dir()):
                continue
            try:
                node_id = int(entry.name.removeprefix("node"))
            except ValueError:
                continue
            cpus = cls._read_cpulist(entry / "cpulist")
            memory = cls._read_meminfo(entry / "meminfo")
            nodes.append(NumaNode(node_id=node_id, cpus=cpus, memory_mib=memory))
        return cls(nodes)

    @staticmethod
    def _read_cpulist(path: Path) -> tuple[int, ...]:
        """Parse a sysfs cpulist such as ``0-3,8``."""
        try:
            text = path.read_text(encoding="utf-8").strip()
        except OSError:
            return ()
        cpus: list[int] = []
        for part in text.split(","):
            if "-" in part:
                low, high = part.split("-", 1)
                try:
                    cpus.extend(range(int(low), int(high) + 1))
                except ValueError:
                    continue
            else:
                try:
                    cpus.append(int(part))
                except ValueError:
                    continue
        return tuple(cpus)

    @staticmethod
    def _read_meminfo(path: Path) -> float:
        """Parse a node meminfo into MiB."""
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            return 0.0
        for line in text.splitlines():
            if line.startswith("MemTotal:"):
                tokens = line.split()
                if len(tokens) >= 2:
                    try:
                        return float(tokens[1]) / 1024.0
                    except ValueError:
                        return 0.0
        return 0.0

    @property
    def locality_available(self) -> bool:
        """Return True only when a real multi-node topology is known."""
        return len(self.nodes) > 1

    def placement_cost(self, node_id: int, accessed_nodes: Iterable[int]) -> float:
        """Return the placement cost including the remote-access penalty.

        Cost is the remote-access penalty weighted by the fraction of
        accessed nodes that are remote to the placement node.
        """
        if not self.nodes:
            raise RuntimeError("NUMA topology unavailable; locality cannot be costed")
        if node_id not in {node.node_id for node in self.nodes}:
            raise ValueError(f"unknown NUMA node {node_id}")
        accessed = set(accessed_nodes)
        if not accessed:
            return 0.0
        remote = sum(1 for item in accessed if item != node_id)
        return self.REMOTE_PENALTY * remote / len(accessed)

    def choose_node(self, accessed_nodes: Iterable[int]) -> int | None:
        """Return the node minimizing placement cost, or None without topology."""
        if not self.locality_available:
            return None
        accessed = set(accessed_nodes)
        best_node: int | None = None
        best_cost = math.inf
        for node in self.nodes:
            if not accessed:
                cost = 0.0
            else:
                remote = sum(1 for item in accessed if item != node.node_id)
                cost = self.REMOTE_PENALTY * remote / len(accessed)
            if cost < best_cost:
                best_cost, best_node = cost, node.node_id
        return best_node


@dataclass(frozen=True)
class DataClassProfile:
    """Access and cost profile of one data class (Step 33)."""

    name: str
    access_frequency: float
    decompression_cost: float
    latency_tolerance: float
    durability_requirement: str = "standard"

    def __post_init__(self) -> None:
        """Validate the profile."""
        for name in ("access_frequency", "decompression_cost", "latency_tolerance"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")


class CompressionTierPolicy:
    """Workload-driven compression and tiering decisions (Step 33)."""

    def __init__(self, *, compression_ratio: float = 0.5, ssd_latency_ms: float = 0.1) -> None:
        """Configure expected compression ratio and slower-tier latency."""
        if not 0.0 < compression_ratio < 1.0:
            raise ValueError("compression_ratio must be within (0, 1)")
        self.compression_ratio = compression_ratio
        self.ssd_latency_ms = ssd_latency_ms

    def recommend(self, profile: DataClassProfile) -> "TieringDecision":
        """Recommend in-memory, compressed or SSD-tier storage for one class."""
        if profile.access_frequency <= 0.0:
            return TieringDecision(
                data_class=profile.name, decision="ssd-tier",
                rationale="cold data: no observed access frequency",
            )
        expected_savings = 1.0 - self.compression_ratio
        if profile.latency_tolerance >= 10.0 * profile.decompression_cost:
            decision = "compressed"
            rationale = (
                f"latency tolerance {profile.latency_tolerance:.3f} covers decompression "
                f"cost {profile.decompression_cost:.3f}; saves {expected_savings:.0%}"
            )
        else:
            decision = "in-memory"
            rationale = (
                f"decompression cost {profile.decompression_cost:.3f} dominates latency "
                f"tolerance {profile.latency_tolerance:.3f}"
            )
        return TieringDecision(
            data_class=profile.name, decision=decision, rationale=rationale,
        )


@dataclass(frozen=True)
class TieringDecision:
    """One tiering recommendation with its rationale (Step 33)."""

    data_class: str
    decision: str
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        """Return a detached, JSON-compatible snapshot of the decision."""
        return {
            "data_class": self.data_class, "decision": self.decision,
            "rationale": self.rationale,
        }


@dataclass(frozen=True)
class AcceleratorAllocation:
    """One accelerator memory allocation record (Step 34)."""

    task_id: str
    allocated_mib: float
    timestamp: float


class AcceleratorMemoryManager:
    """Accelerator-aware admission and memory scheduling (Step 34)."""

    def __init__(self, *, capacity_mib: float, transfer_batch_size: float = 64.0) -> None:
        """Configure accelerator memory capacity and batching threshold."""
        if not math.isfinite(capacity_mib) or capacity_mib <= 0.0:
            raise ValueError("capacity_mib must be finite and positive")
        self.capacity_mib = capacity_mib
        self.transfer_batch_size = transfer_batch_size
        self._allocations: dict[str, AcceleratorAllocation] = {}
        self._transfers: list[dict[str, Any]] = []

    def headroom_mib(self) -> float:
        """Return unallocated accelerator memory."""
        used = sum(item.allocated_mib for item in self._allocations.values())
        return max(0.0, self.capacity_mib - used)

    def admit(
        self, task_id: str, *, working_set_mib: float,
    ) -> tuple[str, str]:
        """Admit, defer or reject one task based on accelerator headroom."""
        if not math.isfinite(working_set_mib) or working_set_mib < 0.0:
            raise ValueError("working_set_mib must be finite and non-negative")
        headroom = self.headroom_mib()
        if working_set_mib <= headroom + 1e-9:
            self._allocations[task_id] = AcceleratorAllocation(
                task_id=task_id, allocated_mib=working_set_mib, timestamp=0.0,
            )
            return "ADMIT", f"working set {working_set_mib:.1f} MiB fits headroom"
        if working_set_mib <= self.capacity_mib:
            return (
                "DEFER",
                f"working set {working_set_mib:.1f} MiB exceeds headroom "
                f"{headroom:.1f} MiB; defer until memory frees",
            )
        return (
            "REJECT",
            f"working set {working_set_mib:.1f} MiB exceeds accelerator capacity "
            f"{self.capacity_mib:.1f} MiB",
        )

    def release(self, task_id: str) -> None:
        """Release one task's accelerator memory."""
        self._allocations.pop(task_id, None)

    def plan_transfer(self, task_id: str, volume_mib: float) -> dict[str, Any]:
        """Batch transfers when safe; record the transfer plan (Step 34)."""
        batches = []
        remaining = volume_mib
        while remaining > 0.0:
            batch = min(self.transfer_batch_size, remaining)
            batches.append(batch)
            remaining -= batch
        plan = {
            "task_id": task_id, "volume_mib": volume_mib,
            "batch_count": len(batches), "batch_size_limit": self.transfer_batch_size,
        }
        self._transfers.append(plan)
        return plan


class CxlExtensionPoint:
    """Future CXL device extension interface (Step 35).

    Intentionally minimal: a registry for future device capacity and
    access-cost data. No CXL-specific behaviour is implemented until real
    CXL-enabled hardware is present (spec: "hardware extension interface
    without premature CXL-specific code").
    """

    def __init__(self) -> None:
        """Initialize the empty device registry."""
        self._devices: dict[str, Mapping[str, float]] = {}

    def register_device(self, device_id: str, properties: Mapping[str, float]) -> None:
        """Register one future device with capacity and cost properties."""
        if not isinstance(device_id, str) or not device_id.strip():
            raise ValueError("device_id must be a non-empty string")
        self._devices[device_id] = dict(properties)

    def devices(self) -> Mapping[str, Mapping[str, float]]:
        """Return the registered devices (normally empty today)."""
        return dict(self._devices)

    def placement_cost(self, device_id: str) -> float | None:
        """Return the declared access cost of one device, if registered."""
        device = self._devices.get(device_id)
        if device is None:
            return None
        return float(device.get("access_cost", math.inf))


__all__ = [
    "WorkingSetEstimator", "WorkingSetEstimate", "TieredCache", "CachedEntry",
    "LRUReplacer", "CLOCKReplacer", "MultiGenerationReplacer", "DamonAdapter",
    "DamonReading", "NumaTopology", "NumaNode", "CompressionTierPolicy",
    "DataClassProfile", "TieringDecision", "AcceleratorMemoryManager",
    "AcceleratorAllocation", "CxlExtensionPoint",
]
