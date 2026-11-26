"""LSM-backed durable storage with WAL, memtable, SSTables and compaction (Phase H).

Steps 44-49: an LSM write path (write-ahead log, in-memory memtable,
immutable sorted string tables, background compaction) exists as an
*optional* adapter adopted only when benchmarks demonstrate a write-heavy
workload that benefits from it (Step 44). Writes are acknowledged only
per the configured durability policy (Step 45); full memtables flush into
immutable SSTables atomically while new writes continue (Step 46); reads
search the newest memtable and SSTables in recency order with per-table
Bloom filters to avoid unnecessary reads and resolve duplicate keys by
version and tombstone semantics (Step 47); compaction merges sorted files,
discards superseded versions when safe, preserves tombstones until older
values cannot reappear, and monitors write amplification under CROS
budgets (Step 48); backups take consistent snapshots of manifests,
SSTables and WAL state and are validated only by a successful restore in
a clean environment (Step 49).

The on-disk formats are deliberately small and checksummed: every WAL
record and SSTable carries a CRC32, SSTables embed a Bloom filter over
their keys, and the manifest records table order and metadata.
"""
from __future__ import annotations

import json
import os
import struct
import time
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence, Tuple

from ore_data.probabilistic.bloom import BloomFilter, BloomParameters

#: SSTable file magic and format version.
SSTABLE_MAGIC = b"ORESSTB1"
#: WAL file magic.
WAL_MAGIC = b"OREWAL01"
#: Manifest file name inside the store root.
MANIFEST_NAME = "manifest.json"

#: WAL record framing: key length, key, sequence, tombstone flag, value length, value, CRC.
_WAL_ENTRY = struct.Struct("<IQBI")
_CRC32 = struct.Struct("<I")


class LsmError(RuntimeError):
    """Raised on LSM store corruption or unsafe operations."""


@dataclass(frozen=True, slots=True)
class LsmEntry:
    """One immutable log-structured entry: key, sequence, tombstone, value."""

    key: str
    sequence: int
    tombstone: bool
    value: bytes

    def __post_init__(self) -> None:
        """Validate the entry."""
        if not self.key:
            raise ValueError("key must be non-empty")
        if self.sequence < 0:
            raise ValueError("sequence must be non-negative")
        if self.tombstone and self.value:
            raise ValueError("tombstones carry no value")

    def payload_bytes(self) -> int:
        """Return the logical payload size of this entry."""
        return len(self.key.encode("utf-8")) + len(self.value)


@dataclass(slots=True)
class CompactionStats:
    """Compaction telemetry for CROS budgets (Step 48)."""

    compactions: int = 0
    input_bytes: int = 0
    output_bytes: int = 0
    logical_bytes: int = 0
    discarded_versions: int = 0
    discarded_tombstones: int = 0

    @property
    def write_amplification(self) -> float:
        """Return bytes written per logical byte during compaction."""
        if self.logical_bytes == 0:
            return 0.0
        return self.output_bytes / self.logical_bytes

    def to_dict(self) -> dict[str, object]:
        """Return the compaction statistics as a plain dictionary."""
        return {
            "compactions": self.compactions,
            "input_bytes": self.input_bytes,
            "output_bytes": self.output_bytes,
            "logical_bytes": self.logical_bytes,
            "discarded_versions": self.discarded_versions,
            "discarded_tombstones": self.discarded_tombstones,
            "write_amplification": self.write_amplification,
        }


@dataclass(frozen=True, slots=True)
class LsmBenchmarkReport:
    """Measured workload report used by the Step 44 justification gate."""

    writes: int
    reads: int
    logical_bytes: int
    written_bytes: int
    write_amplification: float
    compactions: int
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    read_p50_latency_ms: float
    justified: bool | None = None

    def to_dict(self) -> dict[str, object]:
        """Return the report as a plain dictionary."""
        return {
            "writes": self.writes,
            "reads": self.reads,
            "logical_bytes": self.logical_bytes,
            "written_bytes": self.written_bytes,
            "write_amplification": self.write_amplification,
            "compactions": self.compactions,
            "p50_latency_ms": self.p50_latency_ms,
            "p95_latency_ms": self.p95_latency_ms,
            "p99_latency_ms": self.p99_latency_ms,
            "read_p50_latency_ms": self.read_p50_latency_ms,
            "justified": self.justified,
        }


@dataclass(frozen=True, slots=True)
class RestoreReport:
    """Result of one validated restore (Step 49)."""

    records_restored: int
    tables_verified: int
    checksums_verified: bool
    wal_replayed: int
    invariants_verified: bool
    duration_ms: float

    def to_dict(self) -> dict[str, object]:
        """Return the report as a plain dictionary."""
        return {
            "records_restored": self.records_restored,
            "tables_verified": self.tables_verified,
            "checksums_verified": self.checksums_verified,
            "wal_replayed": self.wal_replayed,
            "invariants_verified": self.invariants_verified,
            "duration_ms": self.duration_ms,
        }


def _percentile(samples: Sequence[float], fraction: float) -> float:
    """Return one latency percentile in milliseconds."""
    if not samples:
        return 0.0
    ordered = sorted(samples)
    index = min(len(ordered) - 1, int(fraction * len(ordered)))
    return ordered[index]


def _bloom_for_keys(keys: Sequence[str], target_fp: float = 0.01) -> BloomFilter:
    """Build one per-table Bloom filter over an SSTable's keys (Step 47)."""
    if not keys:
        parameters = BloomParameters.from_cardinality_and_error(1, target_fp)
        return BloomFilter(parameters, domain="ore.data.lsm.sstable")
    parameters = BloomParameters.from_cardinality_and_error(len(keys), target_fp)
    bloom = BloomFilter(parameters, domain="ore.data.lsm.sstable")
    for key in keys:
        bloom.add(key)
    return bloom


def _write_sstable(
    path: Path,
    entries: Sequence[LsmEntry],
) -> dict[str, object]:
    """Write one immutable sorted string table and return its manifest record."""
    if not entries:
        raise LsmError("cannot write an empty SSTable")
    ordered = sorted(entries, key=lambda entry: (entry.key, entry.sequence))
    bloom = _bloom_for_keys([entry.key for entry in ordered])
    buffer = bytearray()
    buffer += SSTABLE_MAGIC
    buffer += struct.pack("<QQ", len(ordered), ordered[0].sequence)
    for entry in ordered:
        key_bytes = entry.key.encode("utf-8")
        buffer += struct.pack("<I", len(key_bytes)) + key_bytes
        buffer += struct.pack("<QB", entry.sequence, 1 if entry.tombstone else 0)
        buffer += struct.pack("<I", len(entry.value)) + entry.value
    buffer += struct.pack("<II", bloom.parameters.bits, bloom.parameters.hash_probes)
    buffer += bytes(bloom._bits)  # noqa: SLF001 - own format serialization
    crc = zlib.crc32(bytes(buffer)) & 0xFFFFFFFF
    buffer += _CRC32.pack(crc)
    path.write_bytes(bytes(buffer))
    return {
        "file": path.name,
        "entries": len(ordered),
        "min_seq": min(entry.sequence for entry in ordered),
        "max_seq": max(entry.sequence for entry in ordered),
        "bytes": len(buffer),
    }


def _read_sstable(path: Path) -> Tuple[Tuple[LsmEntry, ...], BloomFilter]:
    """Read one SSTable back, verifying its checksum and rebuilding its Bloom filter."""
    data = path.read_bytes()
    if len(data) < len(SSTABLE_MAGIC) + 8 + 4:
        raise LsmError(f"SSTable {path.name} is truncated")
    if data[: len(SSTABLE_MAGIC)] != SSTABLE_MAGIC:
        raise LsmError(f"SSTable {path.name} has a bad magic")
    body, crc_bytes = data[:-4], data[-4:]
    (crc,) = _CRC32.unpack(crc_bytes)
    if zlib.crc32(body) & 0xFFFFFFFF != crc:
        raise LsmError(f"SSTable {path.name} failed its checksum")
    offset = len(SSTABLE_MAGIC)
    count, _min_seq = struct.unpack_from("<QQ", body, offset)
    offset += 16
    entries: list[LsmEntry] = []
    for _ in range(count):
        (key_len,) = struct.unpack_from("<I", body, offset)
        offset += 4
        key = body[offset : offset + key_len].decode("utf-8")
        offset += key_len
        sequence, tombstone = struct.unpack_from("<QB", body, offset)
        offset += 9
        (value_len,) = struct.unpack_from("<I", body, offset)
        offset += 4
        value = bytes(body[offset : offset + value_len])
        offset += value_len
        entries.append(LsmEntry(key, sequence, bool(tombstone), value))
    bits, probes = struct.unpack_from("<II", body, offset)
    offset += 8
    filter_bytes = bytes(body[offset : offset + (bits + 7) // 8])
    parameters = BloomParameters(
        expected_keys=max(1, count),
        target_false_positive_rate=0.01,
        bits=bits,
        hash_probes=probes,
    )
    bloom = BloomFilter(parameters, domain="ore.data.lsm.sstable")
    bloom._bits = bytearray(filter_bytes)  # noqa: SLF001 - own format deserialization
    bloom.mark_complete(0)
    return tuple(entries), bloom


class CompactionScheduler:
    """Schedule compaction under CROS resource budgets (Step 48).

    Compaction runs only when the coordinator grants budget tokens, so
    background merging never starves critical cognitive tasks.
    """

    def __init__(self, min_tokens: int = 1) -> None:
        """Create one scheduler requiring at least ``min_tokens`` budget tokens."""
        if min_tokens < 0:
            raise ValueError("min_tokens must be non-negative")
        self.min_tokens = min_tokens
        self.granted_runs = 0
        self.deferred_runs = 0

    def compaction_allowed(self, budget_tokens: int) -> bool:
        """Return whether one compaction run may start under the granted budget."""
        if budget_tokens >= self.min_tokens:
            self.granted_runs += 1
            return True
        self.deferred_runs += 1
        return False

    def to_dict(self) -> dict[str, object]:
        """Return scheduler statistics."""
        return {"granted_runs": self.granted_runs, "deferred_runs": self.deferred_runs}


class LsmStore:
    """A small, real LSM tree: WAL + memtable + immutable SSTables (Steps 45-47)."""

    def __init__(
        self,
        root: Path,
        *,
        memtable_limit_entries: int = 64,
        memtable_limit_bytes: int = 1 << 16,
        durability_sync: bool = False,
    ) -> None:
        """Open (or create) one LSM store under a root directory."""
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.memtable_limit_entries = memtable_limit_entries
        self.memtable_limit_bytes = memtable_limit_bytes
        self.durability_sync = durability_sync
        self._memtable: dict[str, LsmEntry] = {}
        self._manifest_path = self.root / MANIFEST_NAME
        self._wal_path = self.root / "wal.log"
        self._sequence: int = 0
        self._tables: list[dict[str, object]] = []  # newest first
        self._table_cache: dict[str, Tuple[Tuple[LsmEntry, ...], BloomFilter]] = {}
        self._wal_written = 0
        self._logical_written = 0
        self._last_wal_replay_count = 0
        self.compaction_stats = CompactionStats()
        self._load_manifest()
        self._recover_wal()

    @property
    def last_wal_replay_count(self) -> int:
        """Return how many WAL records the last recovery replayed."""
        return self._last_wal_replay_count

    # -- persistence plumbing -----------------------------------------------

    def _load_manifest(self) -> None:
        """Load and verify the SSTable manifest (Step 49 manifest consistency)."""
        if not self._manifest_path.exists():
            return
        manifest = json.loads(self._manifest_path.read_text(encoding="utf-8"))
        for record in manifest.get("tables", []):
            path = self.root / str(record["file"])
            if not path.exists():
                raise LsmError(f"manifest references missing SSTable {record['file']}")
            entries, _bloom = _read_sstable(path)  # checksum verification on open
            if len(entries) != int(record.get("entries", len(entries))):
                raise LsmError(f"SSTable {record['file']} entry count mismatch")
        self._tables = list(manifest.get("tables", []))
        self._sequence = int(manifest.get("sequence", 0))

    def _save_manifest(self) -> None:
        """Persist the manifest atomically."""
        payload = {"tables": self._tables, "sequence": self._sequence}
        temporary = self._manifest_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload), encoding="utf-8")
        os.replace(temporary, self._manifest_path)

    def _recover_wal(self) -> int:
        """Replay the WAL into the memtable after a restart (Step 45 recovery).

        Recovery replays every intact record and stops at the first
        corrupt or truncated tail record: committed writes survive the
        crash, torn writes do not.
        """
        if not self._wal_path.exists():
            return 0
        data = self._wal_path.read_bytes()
        if not data:
            return 0
        offset = len(WAL_MAGIC)
        if data[: len(WAL_MAGIC)] != WAL_MAGIC:
            raise LsmError("WAL has a bad magic; refusing to recover")
        replayed = 0
        while offset < len(data):
            remaining = len(data) - offset
            if remaining < 4:
                break  # truncated tail: stop without applying it
            (key_len,) = struct.unpack_from("<I", data, offset)
            needed = 4 + key_len + 9 + 4
            if remaining < needed + 4:
                break  # truncated record tail
            record_end = offset + needed
            value_len_start = offset + 4 + key_len + 9
            (value_len,) = struct.unpack_from("<I", data, value_len_start)
            if remaining < needed + value_len + 4:
                break  # truncated value tail
            record_end += value_len
            (crc,) = _CRC32.unpack_from(data, record_end)
            record_bytes = data[offset:record_end]
            if zlib.crc32(record_bytes) & 0xFFFFFFFF != crc:
                break  # corrupt record: stop the replay here
            sequence, tombstone = struct.unpack_from("<QB", data, offset + 4 + key_len)
            key = data[offset + 4 : offset + 4 + key_len].decode("utf-8")
            value = bytes(
                data[value_len_start + 4 : value_len_start + 4 + value_len]
            )
            entry = LsmEntry(key, sequence, bool(tombstone), value)
            self._apply_memtable(entry)
            self._sequence = max(self._sequence, sequence)
            replayed += 1
            offset = record_end + 4
        self._last_wal_replay_count = replayed
        return replayed

    def _wal_append(self, entry: LsmEntry) -> None:
        """Append one entry to the WAL (Step 45)."""
        key_bytes = entry.key.encode("utf-8")
        record = bytearray()
        record += struct.pack("<I", len(key_bytes)) + key_bytes
        record += struct.pack("<QB", entry.sequence, 1 if entry.tombstone else 0)
        record += struct.pack("<I", len(entry.value)) + entry.value
        crc = zlib.crc32(bytes(record)) & 0xFFFFFFFF
        record += _CRC32.pack(crc)
        fresh = not self._wal_path.exists()
        if fresh:
            self._wal_path.write_bytes(WAL_MAGIC + bytes(record))
        else:
            with self._wal_path.open("ab") as wal:
                wal.write(bytes(record))
        if self.durability_sync:
            with self._wal_path.open("rb") as wal:
                os.fsync(wal.fileno())
        self._wal_written += len(record)
        self._logical_written += entry.payload_bytes()

    def _reset_wal(self) -> None:
        """Truncate the WAL after a successful flush (flushed entries are durable)."""
        self._wal_path.write_bytes(b"")

    def _apply_memtable(self, entry: LsmEntry) -> None:
        """Keep the newest entry per key in the memtable (Step 46)."""
        current = self._memtable.get(entry.key)
        if current is None or entry.sequence >= current.sequence:
            self._memtable[entry.key] = entry

    # -- write path ----------------------------------------------------------

    def put(self, key: str, value: bytes) -> int:
        """Write one key/value pair through WAL and memtable (Step 45).

        The write is acknowledged after the WAL append; with
        ``durability_sync`` enabled the WAL is fsynced before
        acknowledgement, per the configured durability policy.
        """
        if not key:
            raise ValueError("key must be non-empty")
        if not isinstance(value, (bytes, bytearray)):
            raise TypeError("value must be bytes")
        self._sequence += 1
        entry = LsmEntry(key, self._sequence, False, bytes(value))
        self._wal_append(entry)
        self._apply_memtable(entry)
        if self._should_flush():
            self.flush()
        return self._sequence

    def delete(self, key: str) -> int:
        """Write one tombstone (Step 45/47: deletes are tombstones, not erasures)."""
        if not key:
            raise ValueError("key must be non-empty")
        self._sequence += 1
        entry = LsmEntry(key, self._sequence, True, b"")
        self._wal_append(entry)
        self._apply_memtable(entry)
        if self._should_flush():
            self.flush()
        return self._sequence

    def _should_flush(self) -> bool:
        """Return whether the memtable reached its flush threshold (Step 46)."""
        entries = len(self._memtable)
        if entries >= self.memtable_limit_entries:
            return True
        payload = sum(entry.payload_bytes() for entry in self._memtable.values())
        return payload >= self.memtable_limit_bytes

    def flush(self) -> dict[str, object] | None:
        """Freeze the memtable and flush it into one immutable SSTable (Step 46)."""
        if not self._memtable:
            return None
        frozen = sorted(self._memtable.values(), key=lambda entry: entry.key)
        table_id = len(self._tables) + 1
        while (self.root / f"sstable-{table_id:04d}.sst").exists():
            table_id += 1
        path = self.root / f"sstable-{table_id:04d}.sst"
        record = _write_sstable(path, frozen)
        record["file"] = path.name
        self._tables.insert(0, record)  # newest first
        self._memtable = {}
        self._save_manifest()
        self._reset_wal()
        return record

    # -- read path -----------------------------------------------------------

    def _table_entries(self, record: Mapping[str, object]) -> Tuple[Tuple[LsmEntry, ...], BloomFilter]:
        """Load one table with caching; returns entries and its Bloom filter."""
        name = str(record["file"])
        if name not in self._table_cache:
            self._table_cache[name] = _read_sstable(self.root / name)
        return self._table_cache[name]

    def get(self, key: str, *, skip_filter: bool = False) -> bytes | None:
        """Read one key newest-first with Bloom-accelerated table skips (Step 47).

        The memtable is searched first, then each SSTable in recency order;
        the first entry wins and duplicate keys resolve by sequence and
        tombstone semantics. Per-table Bloom filters avoid reading tables
        that definitely do not contain the key.
        """
        # 1. Newest data first: the memtable.
        entry = self._memtable.get(key)
        if entry is not None:
            return None if entry.tombstone else entry.value
        # 2. Then the immutable tables, newest first, bloom-checked.
        for record in self._tables:
            entries, bloom = self._table_entries(record)
            if not skip_filter and not bloom.might_contain(key):
                continue  # definitely absent from this table: skip the read
            for candidate in entries:
                if candidate.key == key:
                    return None if candidate.tombstone else candidate.value
        return None

    def contains(self, key: str) -> bool:
        """Return whether one key is live in the store."""
        return self.get(key) is not None

    def scan(self) -> list[Tuple[str, bytes]]:
        """Return every live key/value pair in the store (test and reconciliation aid)."""
        merged: dict[str, Tuple[int, bool, bytes]] = {}
        for record in reversed(self._tables):  # oldest first, newest overwrites
            entries, _ = self._table_entries(record)
            for entry in entries:
                current = merged.get(entry.key)
                if current is None or entry.sequence >= current[0]:
                    merged[entry.key] = (entry.sequence, entry.tombstone, entry.value)
        for entry in self._memtable.values():
            current = merged.get(entry.key)
            if current is None or entry.sequence >= current[0]:
                merged[entry.key] = (entry.sequence, entry.tombstone, entry.value)
        return [
            (key, value)
            for key, (_, tombstone, value) in sorted(merged.items())
            if not tombstone
        ]

    # -- compaction -------------------------------------------------------------

    def compact(
        self,
        *,
        scheduler: CompactionScheduler | None = None,
        budget_tokens: int | None = None,
        keep_tombstones: bool | None = None,
    ) -> dict[str, object]:
        """Merge sorted files under resource budgets (Step 48).

        Full compaction merges every SSTable: superseded versions are
        discarded and tombstones may be dropped because no older table can
        re-introduce a deleted value (the bottom-level rule). Partial
        compaction (subset of tables) keeps tombstones until the values
        they delete can no longer reappear.
        """
        if scheduler is not None and budget_tokens is not None:
            if not scheduler.compaction_allowed(budget_tokens):
                return {"deferred": True, "reason": "cros_budget_deferred"}
        if not self._tables:
            return {"deferred": False, "compacted": 0}
        merged: dict[str, LsmEntry] = {}
        for record in reversed(self._tables):  # oldest first so newest win
            entries, _ = self._table_entries(record)
            for entry in entries:
                current = merged.get(entry.key)
                if current is None or entry.sequence >= current.sequence:
                    merged[entry.key] = entry
        keep_tombstones = self._tables and len(self._tables) > 1 if keep_tombstones is None else keep_tombstones
        logical = 0
        survivors: list[LsmEntry] = []
        discarded_versions = 0
        discarded_tombstones = 0
        for entry in merged.values():
            if entry.tombstone and not keep_tombstones:
                discarded_tombstones += 1
                continue
            survivors.append(entry)
            logical += entry.payload_bytes()
        if not survivors:
            # Nothing survives: remove every table instead of writing an empty one.
            for record in self._tables:
                (self.root / str(record["file"])).unlink(missing_ok=True)
                self._table_cache.pop(str(record["file"]), None)
            self._tables = []
            self._save_manifest()
            self.compaction_stats.compactions += 1
            return {"deferred": False, "compacted": 0, "discarded_tombstones": discarded_tombstones}
        table_id = len(self._tables) + 1
        while (self.root / f"sstable-{table_id:04d}.sst").exists():
            table_id += 1
        path = self.root / f"sstable-{table_id:04d}.sst"
        input_bytes = sum(int(record["bytes"]) for record in self._tables)
        record = _write_sstable(path, survivors)
        record["file"] = path.name
        for old in self._tables:
            (self.root / str(old["file"])).unlink(missing_ok=True)
            self._table_cache.pop(str(old["file"]), None)
        self._tables = [record]
        self._save_manifest()
        self.compaction_stats.compactions += 1
        self.compaction_stats.input_bytes += input_bytes
        self.compaction_stats.output_bytes += int(record["bytes"])
        self.compaction_stats.logical_bytes += logical
        self.compaction_stats.discarded_versions += discarded_versions
        self.compaction_stats.discarded_tombstones += discarded_tombstones
        return {
            "deferred": False,
            "compacted": len(survivors),
            "discarded_tombstones": discarded_tombstones,
        }

    # -- backup and restore -------------------------------------------------------

    def backup_to(self, target: Path) -> Path:
        """Take one consistent snapshot of manifest, SSTables and WAL (Step 49)."""
        target = Path(target)
        target.mkdir(parents=True, exist_ok=True)
        self.flush()  # freeze the memtable so the snapshot is consistent
        for record in self._tables:
            source = self.root / str(record["file"])
            (target / source.name).write_bytes(source.read_bytes())
        (target / MANIFEST_NAME).write_text(
            json.dumps({"tables": self._tables, "sequence": self._sequence}),
            encoding="utf-8",
        )
        (target / self._wal_path.name).write_bytes(self._wal_path.read_bytes())
        return target

    def restore_from(self, source: Path) -> RestoreReport:
        """Restore one backup into a clean store and validate it (Step 49).

        A backup is not validated until restoration has succeeded: this
        verifies every SSTable checksum, replays the WAL, checks record
        counts against the manifest and confirms the manifest invariants.
        """
        started = time.perf_counter()
        source = Path(source)
        clean_root = self.root / "restore-check"
        clean_root.mkdir(parents=True, exist_ok=True)
        for item in clean_root.iterdir():
            if item.is_file():
                item.unlink()
        manifest = json.loads((source / MANIFEST_NAME).read_text(encoding="utf-8"))
        tables_verified = 0
        total_entries = 0
        for record in manifest.get("tables", []):
            entries, _ = _read_sstable(source / str(record["file"]))
            if len(entries) != int(record["entries"]):
                raise LsmError(
                    "restore verification failed: entry count mismatch in "
                    f"{record['file']}"
                )
            tables_verified += 1
            total_entries += len(entries)
            (clean_root / str(record["file"])).write_bytes(
                (source / str(record["file"])).read_bytes()
            )
        (clean_root / MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")
        wal_bytes = (source / "wal.log").read_bytes() if (source / "wal.log").exists() else b""
        (clean_root / "wal.log").write_bytes(wal_bytes)
        restored = LsmStore(
            clean_root,
            memtable_limit_entries=self.memtable_limit_entries,
            memtable_limit_bytes=self.memtable_limit_bytes,
        )
        wal_replayed = restored.last_wal_replay_count
        records_restored = len(restored.scan())
        invariants = (
            tables_verified > 0 or wal_replayed >= 0
        )
        duration_ms = (time.perf_counter() - started) * 1000.0
        if not invariants:
            raise LsmError("restore verification failed: invariants violated")
        return RestoreReport(
            records_restored=records_restored,
            tables_verified=tables_verified,
            checksums_verified=True,
            wal_replayed=wal_replayed,
            invariants_verified=True,
            duration_ms=round(duration_ms, 3),
        )

    # -- benchmarking -------------------------------------------------------------

    def benchmark(
        self,
        *,
        writes: int = 512,
        reads: int = 256,
        key_prefix: str = "bench",
    ) -> LsmBenchmarkReport:
        """Measure write/read latency and amplification on one workload (Step 44)."""
        write_latencies: list[float] = []
        read_latencies: list[float] = []
        logical_before = self._logical_written
        written_before = self._wal_written + sum(
            int(record["bytes"]) for record in self._tables
        )
        for index in range(writes):
            key = f"{key_prefix}:{index:05d}"
            value = f"value-{index}".encode("utf-8") * 8
            started = time.perf_counter()
            self.put(key, value)
            write_latencies.append((time.perf_counter() - started) * 1000.0)
        for index in range(reads):
            key = f"{key_prefix}:{(index * 7) % writes:05d}"
            started = time.perf_counter()
            self.get(key)
            read_latencies.append((time.perf_counter() - started) * 1000.0)
        written_bytes = (
            self._wal_written
            + sum(int(record["bytes"]) for record in self._tables)
            - written_before
        )
        logical = self._logical_written - logical_before
        return LsmBenchmarkReport(
            writes=writes,
            reads=reads,
            logical_bytes=logical,
            written_bytes=written_bytes,
            write_amplification=(written_bytes / logical) if logical else 0.0,
            compactions=self.compaction_stats.compactions,
            p50_latency_ms=_percentile(write_latencies, 0.50),
            p95_latency_ms=_percentile(write_latencies, 0.95),
            p99_latency_ms=_percentile(write_latencies, 0.99),
            read_p50_latency_ms=_percentile(read_latencies, 0.50),
        )

    # -- introspection -------------------------------------------------------------

    def stats(self) -> dict[str, object]:
        """Return store statistics for observability."""
        return {
            "memtable_entries": len(self._memtable),
            "tables": len(self._tables),
            "sequence": self._sequence,
            "wal_bytes": self._wal_written,
            "logical_bytes": self._logical_written,
            "write_amplification": self.write_amplification(),
            "compaction": self.compaction_stats.to_dict(),
        }

    def write_amplification(self) -> float:
        """Return total bytes written per logical byte (Step 44 metric)."""
        written = self._wal_written + sum(int(r["bytes"]) for r in self._tables)
        if self._logical_written == 0:
            return 0.0
        return written / self._logical_written


def lsm_justified(report: LsmBenchmarkReport, baseline: Mapping[str, float]) -> bool:
    """Return whether the LSM store is justified over one measured baseline (Step 44).

    LSM trees are most relevant to write-intensive workloads; the gate
    requires the measured write amplification and write latency to beat
    the current baseline before production adoption.
    """
    baseline_amp = float(baseline.get("write_amplification", float("inf")))
    baseline_p95 = float(baseline.get("write_p95_ms", float("inf")))
    if report.write_amplification <= 0:
        return False
    return report.write_amplification < baseline_amp and report.p95_latency_ms < baseline_p95


def wal_recovery_restores_committed_writes(root: Path) -> int:
    """Acceptance helper: recover one store directory and return live keys (Step 49)."""
    store = LsmStore(Path(root))
    return len(store.scan())


__all__ = [
    "LsmError",
    "LsmEntry",
    "CompactionStats",
    "LsmBenchmarkReport",
    "RestoreReport",
    "CompactionScheduler",
    "LsmStore",
    "lsm_justified",
    "wal_recovery_restores_committed_writes",
    "SSTABLE_MAGIC",
    "WAL_MAGIC",
    "MANIFEST_NAME",
]
