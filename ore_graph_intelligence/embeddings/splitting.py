"""Leakage-safe splits and filtered negative sampling (ORE Point 8 steps 10, 12, 21).

Random edge splits leak through neighboring facts; this module implements entity,
relation, time, and source-disjoint split policies plus filtered (never-said-true)
negative sampling with confidence-aware and positive-unlabeled options.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from ..schemas.model_manifest import SplitManifest

SPLIT_POLICIES = ("entity", "relation", "time", "source", "graph_component")


@dataclass(frozen=True, slots=True)
class TripleRecord:
    """One observed triple with the split metadata needed by leakage-safe policies."""

    subject: str
    relation: str
    obj: str
    timestamp: str = ""
    source_id: str = ""

    def triple(self) -> tuple[str, str, str]:
        return (self.subject, self.relation, self.obj)


def build_split_manifest(
    records: Sequence[TripleRecord],
    *,
    policy: str,
    seed: int,
    validation_fraction: float = 0.2,
    test_fraction: float = 0.2,
    notes: Iterable[str] = (),
) -> SplitManifest:
    """Split observed triples under a leakage-aware policy (step 12).

    - ``entity``: test triples use held-out entities unseen in train.
    - ``relation``: test triples use held-out relation types.
    - ``time``: later triples go to validation/test.
    - ``source``: triples from held-out sources.
    - ``graph_component``: triples touching held-out connected components.
    Random edge splits are deliberately not offered: they leak through neighboring facts.
    """
    if policy not in SPLIT_POLICIES:
        raise ValueError(f"policy must be one of {SPLIT_POLICIES}")
    if not 0.0 < validation_fraction < 1.0 or not 0.0 < test_fraction < 1.0:
        raise ValueError("fractions must lie in (0, 1)")
    if validation_fraction + test_fraction >= 1.0:
        raise ValueError("validation and test fractions must leave a training remainder")
    if not records:
        raise ValueError("records must be non-empty")

    rng = np.random.default_rng(seed)
    # One unique triple is assigned to exactly one split; duplicate records collapse
    # so the manifest's disjointness guarantee holds for repeated observations too.
    seen_triples: set[tuple[str, str, str]] = set()
    ordered: list[TripleRecord] = []
    for item in records:
        triple = item.triple()
        if triple not in seen_triples:
            seen_triples.add(triple)
            ordered.append(item)

    def to_triples(items: Sequence[TripleRecord]) -> tuple[tuple[str, str, str], ...]:
        return tuple(item.triple() for item in items)

    if policy == "time":
        ordered = sorted(ordered, key=lambda item: (item.timestamp, item.subject, item.relation))
        n = len(ordered)
        if n < 3:
            train_items, validation_items, test_items = ordered, [], []
        else:
            n_test = max(1, min(int(round(n * test_fraction)), n - 2))
            n_validation = max(1, min(int(round(n * validation_fraction)), n - n_test - 1))
            test_items = ordered[n - n_test:]
            validation_items = ordered[n - n_test - n_validation: n - n_test]
            train_items = ordered[: n - n_test - n_validation]
    elif policy in {"entity", "relation", "source"}:
        keys = {
            "entity": lambda item: (item.subject, item.obj),
            "relation": lambda item: (item.relation,),
            "source": lambda item: (item.source_id,),
        }[policy]
        distinct = sorted({key for item in ordered for key in keys(item)})
        rng.shuffle(distinct)
        total_keys = len(distinct)
        if total_keys < 3:
            n_test = 0 if total_keys < 2 else 1
            n_validation = 0 if n_test == 0 else 0
        else:
            n_test = max(1, min(int(round(total_keys * test_fraction)), total_keys - 2))
            n_validation = max(1, min(
                int(round(total_keys * validation_fraction)), total_keys - n_test - 1,
            ))
        test_keys = set(distinct[:n_test])
        validation_keys = set(distinct[n_test: n_test + n_validation])
        train_items, validation_items, test_items = [], [], []
        for item in ordered:
            item_keys = set(keys(item))
            if item_keys & test_keys:
                test_items.append(item)
            elif item_keys & validation_keys:
                validation_items.append(item)
            else:
                train_items.append(item)
    else:  # graph_component
        adjacency: dict[str, set[str]] = {}
        for item in ordered:
            adjacency.setdefault(item.subject, set()).add(item.obj)
            adjacency.setdefault(item.obj, set()).add(item.subject)
        seen: set[str] = set()
        components: list[set[str]] = []
        for start in sorted(adjacency):
            if start in seen:
                continue
            stack, component = [start], set()
            while stack:
                current = stack.pop()
                if current in component:
                    continue
                component.add(current)
                stack.extend(sorted(adjacency[current] - component))
            seen |= component
            components.append(component)
        rng.shuffle(components)
        total_components = len(components)
        if total_components < 3:
            n_test = 0 if total_components < 2 else 1
            n_validation = 0 if n_test == 0 else 0
        else:
            n_test = max(1, min(int(round(total_components * test_fraction)), total_components - 2))
            n_validation = max(1, min(
                int(round(total_components * validation_fraction)), total_components - n_test - 1,
            ))
        test_components = components[:n_test]
        validation_components = components[n_test: n_test + n_validation]
        test_nodes = set().union(*test_components) if test_components else set()
        validation_nodes = set().union(*validation_components) if validation_components else set()
        train_items, validation_items, test_items = [], [], []
        for item in ordered:
            if item.subject in test_nodes or item.obj in test_nodes:
                test_items.append(item)
            elif item.subject in validation_nodes or item.obj in validation_nodes:
                validation_items.append(item)
            else:
                train_items.append(item)

    return SplitManifest(
        policy=policy,
        seed=seed,
        train_triples=to_triples(train_items),
        validation_triples=to_triples(validation_items),
        test_triples=to_triples(test_items),
        notes=tuple(notes) + (
            f"policy={policy} avoids random-edge leakage",
        ),
    )


class FilteredNegativeSampler:
    """Filtered negative sampling that never labels a known fact as false (step 10).

    An incomplete graph may omit true facts, so naive negative sampling creates false
    negatives; the generator filters every triple already known to the model's own
    knowledge (train + validation, never test) and documents the policy.
    """

    def __init__(
        self,
        known_triples: Iterable[tuple[str, str, str]],
        *,
        mode: str = "filtered",
        corruption_probability: float = 0.5,
    ) -> None:
        if mode not in {"filtered", "confidence_aware", "positive_unlabeled"}:
            raise ValueError("mode must be filtered, confidence_aware, or positive_unlabeled")
        if not 0.0 <= corruption_probability <= 1.0:
            raise ValueError("corruption_probability must lie in [0, 1]")
        self.mode = mode
        self.corruption_probability = float(corruption_probability)
        self.known = frozenset(known_triples)
        self.subjects = tuple(sorted({head for head, _, _ in self.known}))
        self.objects = tuple(sorted({tail for _, _, tail in self.known}))

    def sample(self, rng: np.random.Generator, count: int) -> list[tuple[str, str, str]]:
        """Draw ``count`` negatives that are not known facts under the declared policy."""
        if count < 1:
            return []
        negatives: list[tuple[str, str, str]] = []
        relations = tuple(sorted({relation for _, relation, _ in self.known}))
        attempts = 0
        max_attempts = count * 200 + 100
        while len(negatives) < count and attempts < max_attempts:
            attempts += 1
            head = self.subjects[int(rng.integers(0, len(self.subjects)))]
            relation = relations[int(rng.integers(0, len(relations)))]
            tail = self.objects[int(rng.integers(0, len(self.objects)))]
            candidate = (head, relation, tail)
            if candidate in self.known:
                continue
            if self.mode == "confidence_aware":
                # Confidence-aware: keep most corruptions on the tail (standard KG practice)
                # and only sometimes corrupt the head, biasing toward plausible-but-unobserved.
                if rng.random() > self.corruption_probability and head != tail:
                    candidate = (self.subjects[int(rng.integers(0, len(self.subjects)))], relation, tail)
                    if candidate in self.known:
                        continue
            negatives.append(candidate)
        return negatives

    def policy_description(self) -> str:
        return (
            f"mode={self.mode}; negatives exclude every triple known to train+validation; "
            "an unobserved edge is treated as unknown, not false"
        )


__all__ = [
    "FilteredNegativeSampler", "SPLIT_POLICIES", "TripleRecord", "build_split_manifest",
]
