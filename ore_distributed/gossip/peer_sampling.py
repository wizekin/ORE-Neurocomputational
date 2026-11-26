"""Bounded randomized peer sampling with diversity constraints (Step 31).

The peer set stays randomized, bounded, and diverse: no single static hub, and
no peer sample drawn entirely from one failure domain. Eligibility is filtered
by authenticated membership information rather than raw discovery.
"""
from __future__ import annotations

import random
from typing import Iterable, Mapping

from ore_distributed.membership.registry import NodeHealth, NodeRegistry


class PeerSamplingService:
    """Select gossip peers with fanout, diversity, and authentication filters."""

    def __init__(
        self,
        self_node: str,
        *,
        max_peers: int = 6,
        per_domain_limit: int = 2,
        fanout: int = 2,
        seed: int = 31,
    ) -> None:
        """Create a sampler with explicit fanout and diversity bounds."""
        if not self_node.strip():
            raise ValueError("self_node must be non-empty")
        if max_peers < 1 or per_domain_limit < 1 or fanout < 1:
            raise ValueError("max_peers, per_domain_limit, and fanout must be positive")
        self.self_node = self_node
        self.max_peers = int(max_peers)
        self.per_domain_limit = int(per_domain_limit)
        self.fanout = int(fanout)
        self._rng = random.Random(seed)

    def eligible_peers(self, registry: NodeRegistry) -> tuple[str, ...]:
        """Return authenticated, non-removed peers excluding this node."""
        eligible = []
        for record in registry.authenticated():
            if record.node_id == self.self_node:
                continue
            if record.health in {NodeHealth.REMOVED}:
                continue
            eligible.append(record.node_id)
        return tuple(sorted(eligible))

    def sample_view(self, registry: NodeRegistry) -> tuple[str, ...]:
        """Build one bounded randomized peer view honoring domain diversity."""
        eligible = list(self.eligible_peers(registry))
        self._rng.shuffle(eligible)
        view: list[str] = []
        per_domain: dict[str, int] = {}
        for node_id in eligible:
            record = registry.get(node_id)
            domain = record.failure_domain if record else "default"
            if per_domain.get(domain, 0) >= self.per_domain_limit:
                continue
            view.append(node_id)
            per_domain[domain] = per_domain.get(domain, 0) + 1
            if len(view) >= self.max_peers:
                break
        return tuple(view)

    def select_fanout(self, view: Iterable[str]) -> tuple[str, ...]:
        """Pick the randomized gossip fanout subset from a peer view."""
        candidates = [item for item in view if item != self.self_node]
        if not candidates:
            return ()
        take = min(self.fanout, len(candidates))
        return tuple(sorted(self._rng.sample(candidates, take)))

    def diversity_report(self, view: Iterable[str], registry: NodeRegistry) -> dict[str, int]:
        """Count peers per failure domain to expose single-domain concentration."""
        report: dict[str, int] = {}
        for node_id in view:
            record = registry.get(node_id)
            domain = record.failure_domain if record else "default"
            report[domain] = report.get(domain, 0) + 1
        return report

    def avoid_static_hub(self, previous_view: Iterable[str], registry: NodeRegistry) -> bool:
        """Return whether a new sample avoids repeating one static hub set."""
        previous = tuple(previous_view)
        if len(previous) < 2:
            return True
        fresh = self.sample_view(registry)
        return set(fresh) != set(previous)


def fanout_spread_estimate(
    population: int, fanout: int, rounds: int
) -> int:
    """Estimate dissemination reach after repeated gossip rounds (Step 32).

    With randomized fanout ``f`` from ``n`` peers and repeated rounds, gossip
    spreads updates without every node contacting every other node; the
    exact dissemination time depends on membership, fanout, loss, and churn.
    """
    if population <= 1:
        return population
    if fanout < 1 or rounds < 0:
        raise ValueError("fanout must be positive and rounds non-negative")
    informed = 1
    for _ in range(int(rounds)):
        newly_informed = informed * (1 + fanout) * (1 - informed / population)
        informed = min(population, informed + newly_informed)
    return int(informed)


def gossip_message_budget(rounds: int, fanout: int, payload_bytes: int) -> int:
    """Return the bounded byte budget gossip is allowed per cycle (Step 32)."""
    if rounds < 0 or fanout < 0 or payload_bytes < 0:
        raise ValueError("budget inputs must be non-negative")
    return int(rounds * fanout * payload_bytes)


__all__ = [
    "PeerSamplingService",
    "fanout_spread_estimate",
    "gossip_message_budget",
]
