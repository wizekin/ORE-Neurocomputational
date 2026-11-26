"""State-based CRDT types with documented merge algebra (Steps 23 and 24).

Every type here is a join-semilattice: the merge operator is associative,
commutative, and idempotent, so duplicate, reordered, and repeated state
delivery converges. Each type documents its assumptions and its tombstone or
garbage-collection behavior; records without a valid merge algebra must use
explicit conflict records or authoritative serialization instead (Step 23).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping
import uuid

#: Assumption text recorded for every CRDT kind (Step 23 documentation duty).
CRDTSpec = {
    "gset": {
        "merge_algebra": "set union (join on the subset order)",
        "assumptions": "grow-only: once added, elements are never removed",
        "tombstones": "none required: additions are monotone",
        "garbage_collection": "not applicable",
    },
    "orset": {
        "merge_algebra": "union of add tags and remove tombstones; element visible "
        "iff at least one live tag remains",
        "assumptions": "observed-remove semantics; unique add tags per replica; "
        "removes only observe tags visible at removal time",
        "tombstones": "remove tombstones persist until causally stable",
        "garbage_collection": "bounded by the tombstone collector after stability",
    },
    "pncounter": {
        "merge_algebra": "componentwise max over per-replica increments",
        "assumptions": "increments are local and monotone per replica",
        "tombstones": "none: counters are monotone states",
        "garbage_collection": "not applicable",
    },
    "lww_register": {
        "merge_algebra": "last-writer-wins by (timestamp, writer) tiebreak",
        "assumptions": "timestamps are HLC values; concurrent writers are totally "
        "ordered by the tiebreak, which is a policy choice, not truth",
        "tombstones": "none: the register holds exactly one value",
        "garbage_collection": "not applicable",
    },
}


def _fresh_tag(prefix: str) -> str:
    """Return a collision-resistant unique tag for one add event."""
    return f"{prefix}:{uuid.uuid4().hex}"


class GSet:
    """Grow-only set: merge is union; elements can never be removed."""

    def __init__(self, elements: Iterable[Any] = ()) -> None:
        """Create a grow-only set from initial elements."""
        self._elements: set[Any] = set(elements)

    def add(self, element: Any) -> str:
        """Add one element; returns the operation tag (no removal exists)."""
        self._elements.add(element)
        return "add"

    def merge(self, other: "GSet") -> "GSet":
        """Merge by union, mutating self and returning it."""
        if not isinstance(other, GSet):
            raise TypeError("merge expects a GSet")
        self._elements |= other._elements
        return self

    def value(self) -> frozenset[Any]:
        """Return the current elements."""
        return frozenset(self._elements)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, GSet) and self._elements == other._elements

    def __hash__(self) -> int:
        return hash(frozenset(self._elements))

    def clone(self) -> "GSet":
        """Return an independent copy."""
        return GSet(self._elements)

    def state(self) -> dict[str, Any]:
        """Return the serializable state."""
        return {"kind": "gset", "elements": sorted(self._elements, key=repr)}


class ORSet:
    """Observed-remove set: adds win over removes that did not observe them.

    Every add carries a unique tag; a remove tombstones exactly the tags
    observed at removal time; merge unions tags and tombstones; an element is
    present iff at least one live tag remains.
    """

    def __init__(
        self,
        tags: Mapping[Any, set[str]] | None = None,
        tombstones: Mapping[str, set[Any]] | None = None,
        node_id: str = "local",
    ) -> None:
        """Create an OR-set from serialized state."""
        self._tags: dict[Any, set[str]] = {element: set(tag_set) for element, tag_set in (tags or {}).items()}
        self._tombstones: dict[str, set[Any]] = {tag: set(elements) for tag, elements in (tombstones or {}).items()}
        self._node_id = node_id

    def add(self, element: Any) -> str:
        """Add one element with a fresh unique tag; returns the tag."""
        tag = _fresh_tag(self._node_id)
        self._tags.setdefault(element, set()).add(tag)
        return tag

    def remove(self, element: Any) -> frozenset[str]:
        """Tombstone the currently observed tags for one element."""
        observed = frozenset(self._tags.get(element, ()))
        if not observed:
            return frozenset()
        for tag in observed:
            self._tombstones.setdefault(tag, set()).add(element)
            self._tags.get(element, set()).discard(tag)
        return observed

    def _live_tags(self, element: Any) -> set[str]:
        """Return tags for an element that are not tombstoned."""
        return {
            tag for tag in self._tags.get(element, ())
            if element not in self._tombstones.get(tag, ())
        }

    def contains(self, element: Any) -> bool:
        """Return whether at least one live tag remains for the element."""
        return bool(self._live_tags(element))

    def value(self) -> frozenset[Any]:
        """Return exactly the elements with at least one live tag."""
        return frozenset(element for element in self._tags if self.contains(element))

    def merge(self, other: "ORSet") -> "ORSet":
        """Merge tags and tombstones componentwise, mutating self."""
        if not isinstance(other, ORSet):
            raise TypeError("merge expects an ORSet")
        for element, tags in other._tags.items():
            self._tags.setdefault(element, set()).update(tags)
        for tag, elements in other._tombstones.items():
            self._tombstones.setdefault(tag, set()).update(elements)
        return self

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ORSet):
            return False
        return self.value() == other.value() and self._tombstones == other._tombstones

    def clone(self) -> "ORSet":
        """Return an independent copy including tombstones."""
        return ORSet(self._tags, self._tombstones, self._node_id)

    def tombstone_count(self) -> int:
        """Return the number of tombstone records held."""
        return len(self._tombstones)

    def collect_tombstones(self, stable_tags: Iterable[str]) -> int:
        """Drop tombstones whose causal stability is proven; returns dropped count.

        Deletion metadata is removed only when every replica that could
        resurrect the deleted value has observed the deletion (Step 29).
        """
        stable = set(stable_tags)
        dropped = 0
        for tag in stable:
            if tag in self._tombstones:
                dropped += len(self._tombstones.pop(tag))
        return dropped

    def state(self) -> dict[str, Any]:
        """Return the serializable state including tombstones."""
        return {
            "kind": "orset",
            "tags": {element: sorted(tags) for element, tags in self._tags.items()},
            "tombstones": {tag: sorted(elements) for tag, elements in self._tombstones.items()},
        }


class PNCounter:
    """Positive-negative counter merging by componentwise maximum."""

    def __init__(
        self,
        increments: Mapping[str, int] | None = None,
        decrements: Mapping[str, int] | None = None,
    ) -> None:
        """Create a counter from per-replica monotone components."""
        self._increments: dict[str, int] = dict(increments or {})
        self._decrements: dict[str, int] = dict(decrements or {})
        if any(v < 0 for v in self._increments.values()) or any(v < 0 for v in self._decrements.values()):
            raise ValueError("counter components must be non-negative")

    def increment(self, replica_id: str, amount: int = 1) -> None:
        """Increment the counter for one replica."""
        if amount < 0:
            raise ValueError("use decrement for negative amounts")
        self._increments[replica_id] = self._increments.get(replica_id, 0) + amount

    def decrement(self, replica_id: str, amount: int = 1) -> None:
        """Decrement the counter for one replica."""
        if amount < 0:
            raise ValueError("use increment for positive amounts")
        self._decrements[replica_id] = self._decrements.get(replica_id, 0) + amount

    def value(self) -> int:
        """Return the logical counter value."""
        return sum(self._increments.values()) - sum(self._decrements.values())

    def merge(self, other: "PNCounter") -> "PNCounter":
        """Merge by componentwise max, mutating self."""
        if not isinstance(other, PNCounter):
            raise TypeError("merge expects a PNCounter")
        for replica, amount in other._increments.items():
            self._increments[replica] = max(self._increments.get(replica, 0), amount)
        for replica, amount in other._decrements.items():
            self._decrements[replica] = max(self._decrements.get(replica, 0), amount)
        return self

    def __eq__(self, other: object) -> bool:
        return isinstance(other, PNCounter) and (
            self._increments == other._increments and self._decrements == other._decrements
        )

    def clone(self) -> "PNCounter":
        """Return an independent copy."""
        return PNCounter(self._increments, self._decrements)

    def state(self) -> dict[str, Any]:
        """Return the serializable state."""
        return {
            "kind": "pncounter",
            "increments": dict(sorted(self._increments.items())),
            "decrements": dict(sorted(self._decrements.items())),
        }


@dataclass(frozen=True, slots=True)
class LWWRegister:
    """Last-writer-wins register ordered by (timestamp, writer) tiebreak.

    The tiebreak is a deterministic policy for convergence, not a claim about
    which write is semantically correct: semantic conflicts stay explicit in
    :class:`ore_distributed.replication.conflicts.SemanticConflictRecord`.
    """

    value: Any = None
    timestamp: tuple[int, int, str] = (0, 0, "")
    writer: str = ""

    def write(self, new_value: Any, hlc: tuple[int, int, str]) -> "LWWRegister":
        """Return a new register state if the write wins the LWW order."""
        writer = hlc[2]
        if hlc > self.timestamp or (hlc == self.timestamp and writer > self.writer):
            return LWWRegister(value=new_value, timestamp=hlc, writer=writer)
        return self

    def clone(self) -> "LWWRegister":
        """Return an independent copy (the register is immutable)."""
        return LWWRegister(self.value, self.timestamp, self.writer)

    def merge(self, other: "LWWRegister") -> "LWWRegister":
        """Merge by LWW order; returns the winner (no mutation)."""
        if not isinstance(other, LWWRegister):
            raise TypeError("merge expects an LWWRegister")
        if other.timestamp > self.timestamp:
            return other
        if other.timestamp < self.timestamp:
            return self
        return other if other.writer > self.writer else self

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, LWWRegister)
            and self.value == other.value
            and self.timestamp == other.timestamp
            and self.writer == other.writer
        )

    def state(self) -> dict[str, Any]:
        """Return the serializable state."""
        return {
            "kind": "lww_register",
            "value": self.value,
            "timestamp": list(self.timestamp),
            "writer": self.writer,
        }


@dataclass(frozen=True, slots=True)
class MergeLawReport:
    """Results of property checks over a merge operator (Step 24)."""

    associative: bool
    commutative: bool
    idempotent: bool
    states_checked: int
    violations: tuple[str, ...] = ()


def check_merge_laws(
    states: Iterable[Any],
    merge: Any = None,
) -> MergeLawReport:
    """Property-test associativity, commutativity, and idempotence (Step 24).

    ``states`` are CRDT state objects; each must expose ``clone()`` and a
    mutating ``merge(other)`` returning ``self``. The checks are
    merge(merge(a,b),c) == merge(a,merge(b,c)), merge(a,b) == merge(b,a),
    and merge(a,a) == a.
    """
    items = list(states)
    if len(items) < 1:
        raise ValueError("property checks require at least one state")
    violations: list[str] = []

    def merge_pair(a: Any, b: Any) -> Any:
        """Merge b into a clone of a, honoring the merge's returned state."""
        result = a.clone()
        return result.merge(b.clone())

    checked = 0
    for i, a in enumerate(items):
        # Idempotence: merge(a, a) == a.
        if merge_pair(a, a) != a:
            violations.append(f"idempotence violated at index {i}")
        checked += 1
        for j, b in enumerate(items):
            # Commutativity: merge(a, b) == merge(b, a).
            if merge_pair(a, b) != merge_pair(b, a):
                violations.append(f"commutativity violated at pair ({i}, {j})")
            checked += 1
            for c in items:
                # Associativity: merge(merge(a,b),c) == merge(a,merge(b,c)).
                left = merge_pair(merge_pair(a, b), c)
                right = merge_pair(a, merge_pair(b, c))
                if left != right:
                    violations.append(f"associativity violated at triple ({i}, {j}, ...)")
                checked += 1

    return MergeLawReport(
        associative=not any("associativity" in v for v in violations),
        commutative=not any("commutativity" in v for v in violations),
        idempotent=not any("idempotence" in v for v in violations),
        states_checked=checked,
        violations=tuple(violations),
    )


def merge_semilattice_documented(kind: str) -> dict[str, str]:
    """Return the documented merge algebra for a CRDT kind (Step 24)."""
    try:
        return dict(CRDTSpec[kind])
    except KeyError as exc:
        raise ValueError(
            f"kind {kind!r} has no documented merge algebra; use explicit conflict "
            "records or authoritative serialization instead (Step 23)"
        ) from exc


__all__ = [
    "CRDTSpec",
    "GSet",
    "ORSet",
    "PNCounter",
    "LWWRegister",
    "MergeLawReport",
    "check_merge_laws",
    "merge_semilattice_documented",
]
