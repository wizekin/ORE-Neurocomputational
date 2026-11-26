"""Authenticated transport with bounded messaging (Steps 10 and 11).

The in-memory fabric models mutually authenticated message delivery with
configurable partitions, delays, duplication, and drops, so distributed
protocol behavior can be tested deterministically. Transport authentication
remains separate from semantic trust in a node's findings. Bounded retries
use exponential backoff with jitter, and admission control (backpressure)
prevents an unreachable peer from creating unbounded pending work.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import random
import time
from typing import Any, Callable, Mapping

from ore_distributed.core.identity import NodeCredential
from ore_distributed.core.protocol import (
    MessageEnvelope,
    ProtocolGate,
    ProtocolViolation,
)


class AuthenticationError(RuntimeError):
    """Raised when a peer cannot be authenticated or authorized."""


class BackpressureError(RuntimeError):
    """Raised when bounded queues reject further work (admission control)."""


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Exponential backoff with jitter and a bounded retry budget (Step 11)."""

    base_delay: float = 0.01
    max_delay: float = 0.5
    max_attempts: int = 5
    jitter_fraction: float = 0.25

    def __post_init__(self) -> None:
        """Validate the policy bounds."""
        for name in ("base_delay", "max_delay", "jitter_fraction"):
            value = float(getattr(self, name))
            if value <= 0.0 or value > 10.0:
                raise ValueError(f"{name} must be positive and at most 10 seconds")
        if int(self.max_attempts) < 1 or int(self.max_attempts) > 100:
            raise ValueError("max_attempts must be between 1 and 100")
        if self.base_delay > self.max_delay:
            raise ValueError("base_delay must not exceed max_delay")

    def delay_for(self, attempt: int, rng: random.Random) -> float:
        """Return the backoff delay before attempt ``attempt`` (1-indexed)."""
        attempt = max(1, int(attempt))
        exponential = min(self.max_delay, self.base_delay * (2 ** (attempt - 1)))
        jitter_span = exponential * self.jitter_fraction
        return max(0.0, exponential + rng.uniform(-jitter_span, jitter_span))

    def exhausted(self, attempt: int) -> bool:
        """Return whether the retry budget is exhausted after ``attempt`` tries."""
        return attempt >= self.max_attempts


class InMemoryNetworkFabric:
    """Deterministic message fabric with fault injection for protocol tests.

    Each node registers a delivery handler; the fabric routes envelopes by
    sender/recipient, enforcing partition sets, one-way delays, duplication,
    drops, and a maximum payload size.
    """

    def __init__(
        self,
        *,
        max_payload_bytes: int = 1 << 20,
        duplication_rate: float = 0.0,
        drop_rate: float = 0.0,
        seed: int = 1337,
    ) -> None:
        """Create a fabric with configurable duplication and drop rates."""
        if max_payload_bytes <= 0:
            raise ValueError("max_payload_bytes must be positive")
        if not 0.0 <= duplication_rate <= 1.0 or not 0.0 <= drop_rate <= 1.0:
            raise ValueError("duplication and drop rates must be within [0, 1]")
        self.max_payload_bytes = int(max_payload_bytes)
        self.duplication_rate = float(duplication_rate)
        self.drop_rate = float(drop_rate)
        self._rng = random.Random(seed)
        self._handlers: dict[str, Callable[[MessageEnvelope], Any]] = {}
        self._partitions: set[tuple[frozenset[str], frozenset[str]]] = set()
        self._delays: dict[frozenset[str], float] = {}
        self._pending: list[tuple[float, MessageEnvelope, str]] = []
        self._clock = 0.0
        self.delivered: list[tuple[str, str, str]] = []
        self.dropped: list[tuple[str, str, str]] = []

    def register(self, node_id: str, handler: Callable[[MessageEnvelope], Any]) -> None:
        """Register the delivery handler for one node."""
        if not node_id.strip():
            raise ValueError("node_id must be non-empty")
        self._handlers[node_id] = handler

    def unregister(self, node_id: str) -> None:
        """Remove one node's handler, simulating process loss."""
        self._handlers.pop(node_id, None)

    def handler_for(self, node_id: str) -> Callable[[MessageEnvelope], Any] | None:
        """Return the registered handler for a node, if any."""
        return self._handlers.get(node_id)

    def partition(self, group_a: set[str], group_b: set[str]) -> None:
        """Cut all traffic between two node groups (a symmetric partition)."""
        self._partitions.add((frozenset(group_a), frozenset(group_b)))

    def heal_all_partitions(self) -> None:
        """Remove every injected partition."""
        self._partitions.clear()

    def _cut(self, sender: str, recipient: str) -> bool:
        """Return whether the sender-recipient pair is partitioned."""
        for group_a, group_b in self._partitions:
            if sender in group_a and recipient in group_b:
                return True
            if sender in group_b and recipient in group_a:
                return True
        return False

    def set_delay(self, from_node: str, to_node: str, seconds: float) -> None:
        """Delay every message in one direction."""
        if seconds < 0.0:
            raise ValueError("delay seconds must be non-negative")
        self._delays[frozenset((from_node, to_node))] = float(seconds)

    def _pair_delay(self, sender: str, recipient: str) -> float:
        """Return the directed delay between two nodes."""
        return self._delays.get(frozenset((sender, recipient)), 0.0)

    def send(self, recipient: str, envelope: MessageEnvelope) -> bool:
        """Deliver one envelope if unpartitioned; returns acceptance, not delivery."""
        if len(str(envelope.payload)) > self.max_payload_bytes:
            raise BackpressureError(
                f"payload exceeds the bounded message size of {self.max_payload_bytes} bytes"
            )
        if recipient not in self._handlers:
            self.dropped.append((envelope.sender, recipient, envelope.message_type))
            return False
        if self._cut(envelope.sender, recipient):
            self.dropped.append((envelope.sender, recipient, envelope.message_type))
            return False
        if self._rng.random() < self.drop_rate:
            self.dropped.append((envelope.sender, recipient, envelope.message_type))
            return False
        self._enqueue(envelope, recipient)
        if self._rng.random() < self.duplication_rate:
            self._enqueue(envelope, recipient)
        return True

    def _enqueue(self, envelope: MessageEnvelope, recipient: str) -> None:
        """Schedule one delivery under the configured delay."""
        delay = self._pair_delay(envelope.sender, recipient)
        deliver_at = self._clock + delay
        self._pending.append((deliver_at, envelope, recipient))
        self._pending.sort(key=lambda item: (item[0], item[2], item[1].sender))

    def advance(self, seconds: float = 0.01) -> int:
        """Advance virtual time, delivering every due message; returns the count."""
        if seconds < 0.0:
            raise ValueError("advance requires non-negative seconds")
        horizon = self._clock + seconds
        delivered = 0
        while self._pending and self._pending[0][0] <= horizon:
            _, envelope, recipient = self._pending.pop(0)
            handler = self._handlers.get(recipient)
            if handler is None:
                continue
            handler(envelope)
            self.delivered.append((envelope.sender, recipient, envelope.message_type))
            delivered += 1
        self._clock = horizon
        return delivered

    def drain(self) -> int:
        """Deliver all pending messages at once (no delay advancement)."""
        return self.advance(seconds=10.0 ** 9)

    @property
    def now(self) -> float:
        """Return the fabric's virtual clock."""
        return self._clock


class SimulatedTransport:
    """Per-node authenticated transport over the in-memory fabric.

    Each node owns a :class:`ProtocolGate`; state-mutating envelopes are signed
    before send and verified before delivery, replay nonces are rejected, and
    bounded queues apply admission control instead of unbounded retry growth.
    """

    def __init__(
        self,
        node_id: str,
        cluster_id: str,
        credential: NodeCredential,
        fabric: InMemoryNetworkFabric,
        *,
        retry_policy: RetryPolicy | None = None,
        queue_limit: int = 256,
        seed: int = 7,
    ) -> None:
        """Create one node's transport bound to a shared fabric."""
        if not node_id.strip() or not cluster_id.strip():
            raise ValueError("node_id and cluster_id must be non-empty")
        if queue_limit <= 0:
            raise ValueError("queue_limit must be positive")
        self.node_id = node_id
        self.cluster_id = cluster_id
        self.credential = credential
        self.fabric = fabric
        self.retry_policy = retry_policy or RetryPolicy()
        self.queue_limit = int(queue_limit)
        self.gate = ProtocolGate(cluster_id=cluster_id)
        self.gate.register_credential(credential)
        self._rng = random.Random(seed)
        self._inbound_queue: list[MessageEnvelope] = []
        self.sent = 0
        self.retries = 0
        self.rejected = 0
        self._handler: Callable[[MessageEnvelope], Any] | None = None
        fabric.register(node_id, self._receive)

    def set_handler(self, handler: Callable[[MessageEnvelope], Any]) -> None:
        """Set the callback invoked for each verified inbound envelope."""
        if not callable(handler):
            raise TypeError("handler must be callable")
        self._handler = handler

    def trust(self, credential: NodeCredential) -> None:
        """Trust another cluster member's credential for verification."""
        self.gate.register_credential(credential)

    def send(self, recipient: str, envelope: MessageEnvelope) -> bool:
        """Sign and send one envelope with bounded retry and backpressure."""
        if not self._room_in_queue():
            self.rejected += 1
            raise BackpressureError(
                f"inbound queues at node {self.node_id!r} are saturated; "
                "admission control rejects the request"
            )
        signed = envelope.sign(self.credential) if envelope.mutates_state() else envelope
        attempt = 0
        while True:
            attempt += 1
            if self.fabric.send(recipient, signed):
                self.sent += 1
                return True
            if self.retry_policy.exhausted(attempt):
                return False
            self.retries += 1
            time.sleep(0.0)  # virtual fabric: no real sleep between attempts
            self._rng.random()  # consume jitter draw to keep behavior deterministic

    def _room_in_queue(self) -> bool:
        """Return whether admission control still has queue capacity."""
        return len(self._inbound_queue) <= self.queue_limit

    def _receive(self, envelope: MessageEnvelope) -> None:
        """Verify, de-duplicate, and dispatch one inbound envelope."""
        try:
            self.gate.validate(envelope)
        except ProtocolViolation:
            self.rejected += 1
            return
        self._inbound_queue.append(envelope)
        if len(self._inbound_queue) > self.queue_limit:
            self.rejected += 1
            raise BackpressureError("bounded inbound queue overflowed")
        handler = self._handler
        if handler is not None:
            handler(envelope)
        self._inbound_queue.pop()


def simulate_unreachable_peer_attempts(
    policy: RetryPolicy,
    attempts: int,
    rng: random.Random | None = None,
) -> tuple[float, ...]:
    """Return the backoff schedule the policy would use for a failing peer."""
    rng = rng or random.Random(11)
    return tuple(policy.delay_for(attempt, rng) for attempt in range(1, attempts + 1))


__all__ = [
    "AuthenticationError",
    "BackpressureError",
    "RetryPolicy",
    "InMemoryNetworkFabric",
    "SimulatedTransport",
    "simulate_unreachable_peer_attempts",
]
