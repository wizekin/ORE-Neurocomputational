"""Phase VI, Step 28 - probabilistic data structures for scale, with exact verification.

Bloom filters avoid unnecessary exact lookups when absence is useful,
Count-Min Sketch estimates frequencies, HyperLogLog estimates distinct
counts, and MinHash generates similarity candidates.  False positives are
expected: a positive Bloom-filter answer must be confirmed by exact lookup,
and approximate structures can never become authoritative proof of
existence, identity or correctness.  The Bloom-filter false-positive rate is
``p ~= (1 - e^(-k n / m))^k`` with near-optimal ``k ~= (m/n) ln 2``.
"""
from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

PROBABILISTIC_SCHEMA = "ore.point24.probabilistic.v1"


def _position(item: bytes, index: int, mod: int) -> int:
    digest = hashlib.sha256(item + b":" + struct.pack(">Q", index)).digest()
    return int.from_bytes(digest[:8], "big") % mod


class BloomFilter:
    """A Bloom filter with an explicit, measured false-positive rate."""

    def __init__(self, *, expected_items: int = 1024, false_positive_rate: float = 0.01) -> None:
        if expected_items < 1:
            raise ValueError("expected_items must be at least one")
        if not 0.0 < false_positive_rate < 1.0:
            raise ValueError("false_positive_rate must be in (0, 1)")
        self.expected_items = expected_items
        self.target_rate = false_positive_rate
        optimal_bits = -expected_items * math.log(false_positive_rate) / (math.log(2) ** 2)
        self.size = max(8, int(math.ceil(optimal_bits)))
        self.hash_count = max(1, int(round((self.size / expected_items) * math.log(2))))
        self._bits = bytearray((self.size + 7) // 8)

    def _bit_positions(self, item: bytes) -> Iterable[int]:
        return (_position(item, index, self.size) for index in range(self.hash_count))

    def add(self, item: str | bytes) -> None:
        data = item.encode("utf-8") if isinstance(item, str) else item
        for position in self._bit_positions(data):
            self._bits[position // 8] |= 1 << (position % 8)

    def might_contain(self, item: str | bytes) -> bool:
        """A positive answer is a *candidate*; exact lookup must confirm it."""
        data = item.encode("utf-8") if isinstance(item, str) else item
        return all(
            self._bits[position // 8] & (1 << (position % 8))
            for position in self._bit_positions(data)
        )

    def estimated_false_positive_rate(self, inserted_items: int | None = None) -> float:
        """p ~= (1 - e^(-k n / m))^k from the measured item count."""
        n = self.expected_items if inserted_items is None else inserted_items
        if n <= 0:
            return 0.0
        return (1.0 - math.exp(-self.hash_count * n / self.size)) ** self.hash_count

    def summary(self) -> dict[str, Any]:
        return {
            "schema": PROBABILISTIC_SCHEMA,
            "structure": "bloom_filter",
            "bits": self.size,
            "hash_count": self.hash_count,
            "target_false_positive_rate": self.target_rate,
            "note": "positives are candidates; exact lookup must confirm",
        }


class CountMinSketch:
    """Approximate frequency counts with conservative update and error bounds."""

    def __init__(self, *, width: int = 128, depth: int = 4) -> None:
        if width < 1 or depth < 1:
            raise ValueError("width and depth must be positive")
        self.width = width
        self.depth = depth
        self._table = [[0] * width for _ in range(depth)]

    def add(self, item: str | bytes, count: int = 1) -> None:
        data = item.encode("utf-8") if isinstance(item, str) else item
        for row in range(self.depth):
            self._table[row][_position(data, row, self.width)] += count

    def estimate(self, item: str | bytes) -> int:
        """An upper-biased estimate: count >= estimate - eps with high prob."""
        data = item.encode("utf-8") if isinstance(item, str) else item
        return min(
            self._table[row][_position(data, row, self.width)]
            for row in range(self.depth)
        )

    def summary(self, total_items: int) -> dict[str, Any]:
        return {
            "schema": PROBABILISTIC_SCHEMA,
            "structure": "count_min_sketch",
            "width": self.width,
            "depth": self.depth,
            "epsilon_bound": math.e / self.width,
            "note": "estimates are upper-biased; never authoritative counts",
        }


class HyperLogLog:
    """Distinct-count estimation with the standard HLL bias-corrected mean."""

    def __init__(self, *, precision: int = 8) -> None:
        if not 4 <= precision <= 16:
            raise ValueError("precision (register bits) must be in [4, 16]")
        self.precision = precision
        self.size = 1 << precision
        self._registers = [0] * self.size

    def add(self, item: str | bytes) -> None:
        data = item.encode("utf-8") if isinstance(item, str) else item
        digest = hashlib.sha256(data).digest()
        h = int.from_bytes(digest[:8], "big")
        index = h & (self.size - 1)
        remaining = h >> self.precision
        rank = 1
        while remaining and not (remaining & 1):
            rank += 1
            remaining >>= 1
        if rank > self._registers[index]:
            self._registers[index] = rank

    def estimate(self) -> float:
        alpha = {16: 0.673, 32: 0.697, 64: 0.709}.get(self.size, 0.7213 / (1.0 + 1.079 / self.size))
        zero_registers = self._registers.count(0)
        harmonic = sum(2.0 ** -rank for rank in self._registers)
        if harmonic == 0:
            return 0.0
        raw = alpha * self.size * self.size / harmonic
        if raw <= 2.5 * self.size and zero_registers:
            return self.size * math.log(self.size / zero_registers)
        return raw

    def summary(self) -> dict[str, Any]:
        return {
            "schema": PROBABILISTIC_SCHEMA,
            "structure": "hyper_log_log",
            "precision": self.precision,
            "registers": self.size,
            "note": "distinct-count estimate with standard error ~1.04/sqrt(m)",
        }


class MinHashIndex:
    """MinHash similarity candidate generation with exact re-check policy."""

    def __init__(self, *, hash_count: int = 64, seed: int = 7) -> None:
        if hash_count < 1:
            raise ValueError("hash_count must be positive")
        self.hash_count = hash_count
        self.seed = seed
        self._signatures: dict[str, tuple[int, ...]] = {}

    @staticmethod
    def _shingles(text: str) -> set[str]:
        words = text.casefold().split()
        if len(words) <= 2:
            return {" ".join(words)} if words else set()
        return {" ".join(words[i:i + 2]) for i in range(len(words) - 1)}

    def _signature(self, text: str) -> tuple[int, ...]:
        shingles = self._shingles(text)
        signature = []
        for index in range(self.hash_count):
            values = (
                int.from_bytes(
                    hashlib.sha256(
                        f"{self.seed}:{index}:{shingle}".encode("utf-8")
                    ).digest()[:8], "big"
                )
                for shingle in shingles
            )
            signature.append(min(values) if shingles else 0)
        return tuple(signature)

    def index(self, key: str, text: str) -> None:
        self._signatures[key] = self._signature(text)

    def similarity(self, key_a: str, key_b: str) -> float:
        sig_a = self._signatures.get(key_a)
        sig_b = self._signatures.get(key_b)
        if sig_a is None or sig_b is None:
            raise KeyError("both keys must be indexed")
        matches = sum(1 for a, b in zip(sig_a, sig_b) if a == b)
        return matches / self.hash_count

    def candidates(self, text: str, *, threshold: float = 0.4) -> list[tuple[float, str]]:
        """Return similarity candidates; exact verification stays mandatory."""
        target = self._signature(text)
        scored = []
        for key, signature in self._signatures.items():
            matches = sum(1 for a, b in zip(target, signature) if a == b)
            score = matches / self.hash_count
            if score >= threshold:
                scored.append((score, key))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return scored

    def summary(self) -> dict[str, Any]:
        return {
            "schema": PROBABILISTIC_SCHEMA,
            "structure": "min_hash",
            "hash_count": self.hash_count,
            "indexed_keys": sorted(self._signatures),
            "note": "LSH-style candidate generation only; re-check authoritative records",
        }


__all__ = [
    "PROBABILISTIC_SCHEMA",
    "BloomFilter",
    "CountMinSketch",
    "HyperLogLog",
    "MinHashIndex",
]
