"""Additive secret sharing and Beaver triples over a prime field.

Additive sharing splits a value x into n shares with x = sum(shares) mod p; any
n-1 shares are uniform and reveal nothing about x. Beaver triples (a, b, c = a*b)
are precomputed by a trusted dealer and enable multiplication on shares without
opening the inputs. The trusted-dealer assumption is documented explicitly per
spec step 45: production deployments use a dealerless triple generation or an
MPC protocol with distributed preprocessing.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .fields import PrimeField


@dataclass(frozen=True, slots=True)
class ShareVector:
    """One party's share of a secret vector."""

    party_index: int
    values: tuple[int, ...]


class AdditiveSecretSharer:
    """Create and reconstruct additive shares over the prime field."""

    def __init__(self, field: PrimeField | None = None) -> None:
        self.field = field or PrimeField()

    def share_scalar(self, value: int, party_count: int, rng) -> list[int]:
        """Split one scalar into ``party_count`` uniform shares."""
        if party_count < 2:
            raise ValueError("additive sharing needs at least two parties")
        shares = [self.field.random(rng) for _ in range(party_count - 1)]
        shares.append(self.field.sub(self.field.reduce(value), sum(shares) % self.field.modulus))
        return [self.field.reduce(share) for share in shares]

    def share_vector(self, values: Sequence[int], party_count: int, rng) -> list[ShareVector]:
        """Split a vector into per-party share vectors."""
        columns = [self.share_scalar(value, party_count, rng) for value in values]
        return [
            ShareVector(party_index=party, values=tuple(column[party] for column in columns))
            for party in range(party_count)
        ]

    def reconstruct(self, shares: Sequence[Sequence[int]] | Sequence[ShareVector]) -> list[int]:
        """Sum per-party shares to recover the secret vector."""
        if not shares:
            raise ValueError("reconstruction requires at least one share")
        if isinstance(shares[0], ShareVector):
            vectors = [list(share.values) for share in shares]
        else:
            vectors = [list(share) for share in shares]
        length = len(vectors[0])
        if any(len(vector) != length for vector in vectors):
            raise ValueError("share vectors must have equal length")
        return [
            sum(vector[index] for vector in vectors) % self.field.modulus
            for index in range(length)
        ]


@dataclass(frozen=True, slots=True)
class BeaverTriple:
    """One multiplication triple (a, b, c = a*b) shared across parties."""

    party_count: int
    a_shares: tuple[tuple[int, ...], ...]
    b_shares: tuple[tuple[int, ...], ...]
    c_shares: tuple[tuple[int, ...], ...]


class BeaverTripleFactory:
    """Trusted dealer producing shared multiplication triples (assumption documented)."""

    def __init__(self, field: PrimeField | None = None) -> None:
        self.field = field or PrimeField()
        self.trust_assumption = (
            "trusted-dealer preprocessing: the dealer sees triple values but never "
            "party inputs; dealerless generation is required for stronger models"
        )

    def generate(self, party_count: int, count: int, rng) -> BeaverTriple:
        """Produce ``count`` shared triples for ``party_count`` parties."""
        sharer = AdditiveSecretSharer(self.field)
        a_values = [self.field.random(rng) for _ in range(count)]
        b_values = [self.field.random(rng) for _ in range(count)]
        c_values = [self.field.mul(a, b) for a, b in zip(a_values, b_values)]
        return BeaverTriple(
            party_count=party_count,
            a_shares=tuple(tuple(sharer.share_scalar(a, party_count, rng)) for a in a_values),
            b_shares=tuple(tuple(sharer.share_scalar(b, party_count, rng)) for b in b_values),
            c_shares=tuple(tuple(sharer.share_scalar(c, party_count, rng)) for c in c_values),
        )

    @staticmethod
    def triple_at(triple: BeaverTriple, index: int) -> tuple[list[int], list[int], list[int]]:
        """Extract per-party shares of triple ``index`` (a, b, c = a*b)."""
        return (
            list(triple.a_shares[index]),
            list(triple.b_shares[index]),
            list(triple.c_shares[index]),
        )
