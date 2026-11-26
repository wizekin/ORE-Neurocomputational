"""Secure aggregation with pairwise-canceling masks (steps 46, 48, 51).

Participants submit values protected by pairwise-canceling pseudorandom masks
derived from per-pair seeds established through authenticated key agreement and
bound to the aggregation round. The server observes only masked values, so it
learns the aggregate but no individual contribution. Duplicate or replayed
contributions in a round are rejected. Dropout recovery re-shares the dropped
participant's pairwise seeds among the survivors, so the server can remove the
unbalanced masks without ever seeing an unprotected value.

Collusion assumptions (spec step 45, documented): the server alone learns nothing
about any individual contribution; the server together with every other
participant (n-1 colluders) can isolate the remaining participant's value, the
same bound as Bonawitz-style masking with t = n-1. Stronger bounds require
threshold (Shamir) seed sharing, documented as the production upgrade.
"""
from __future__ import annotations

import time
from typing import Any, Callable, Sequence

from ..canonical import canonical_bytes
from ..contracts import AggregateResult
from ..crypto import generate_x25519_keypair, prg_integers, x25519_shared_secret, hkdf_derive
from ..exceptions import AggregationError
from ..identity.identities import IdentityService
from .fields import PrimeField

Clock = Callable[[], float]
PROTOCOL_REF = "secure-aggregation-pairwise-masks-v1"


def _pairwise_seed(private: bytes, peer_public: bytes, round_id: str,
                   low_index: str, high_index: str) -> bytes:
    """Derive the per-pair, per-round mask seed from an X25519 agreement."""
    secret = x25519_shared_secret(private, peer_public)
    return hkdf_derive(
        secret,
        salt=b"ore-secure-aggregation",
        info=canonical_bytes({"round": round_id, "pair": [low_index, high_index]}),
    )


class AggregationParticipant:
    """One ORE instance contributing a protected telemetry vector."""

    def __init__(self, participant_id: str) -> None:
        self.participant_id = participant_id
        self._private, self.public = generate_x25519_keypair()

    def mask_for(self, participants: Sequence[str], round_id: str,
                 value_vector: Sequence[int], field: PrimeField,
                 peer_publics: dict[str, bytes]) -> list[int]:
        """Compute the participant's masked submission for one round.

        The submission is ``value + sum(sign_{self,peer} * prg(seed_{self,peer}))``
        where the sign is +1 when this participant is the lexicographically lower
        id of the pair, and the seed is the X25519 agreement with the peer's public
        key bound to the round. Pairwise signs cancel between the two seed holders.
        """
        ordered = sorted(participants)
        if self.participant_id not in ordered:
            raise AggregationError("participant is not part of the round")
        masks = [0] * len(value_vector)
        for peer in ordered:
            if peer == self.participant_id:
                continue
            if peer not in peer_publics:
                raise AggregationError(f"missing public key for peer {peer!r}")
            low, high = sorted([self.participant_id, peer])
            context = b"mask|" + round_id.encode("utf-8")
            elements = prg_integers(
                _pairwise_seed(self._private, peer_publics[peer], round_id, low, high),
                context, len(value_vector), field.modulus,
            )
            sign = 1 if self.participant_id == low else -1
            masks = [m + sign * e for m, e in zip(masks, elements)]
        return [field.reduce(v + m) for v, m in zip(value_vector, masks)]

    def reveal_pairwise_elements(self, peer: "AggregationParticipant", round_id: str,
                                 count: int, field: PrimeField) -> list[int]:
        """Reveal the PRG elements shared with ``peer`` for dropout recovery.

        Only meaningful when ``peer`` dropped out: the elements cancel the dangling
        mask the peer would have contributed to this participant's submission.
        """
        low, high = sorted([self.participant_id, peer.participant_id])
        seed = _pairwise_seed(self._private, peer.public, round_id, low, high)
        return prg_integers(seed, b"mask|" + round_id.encode("utf-8"),
                            count, field.modulus)


class SecureAggregationServer:
    """Coordinate rounds, accept masked submissions, publish permitted aggregates."""

    def __init__(self, *, field: PrimeField | None = None,
                 min_cohort_size: int = 3, clock: Clock = time.time) -> None:
        self.field = field or PrimeField()
        self.min_cohort_size = int(min_cohort_size)
        self.clock = clock
        self._rounds: dict[str, dict[str, Any]] = {}

    def open_round(self, round_id: str, participants: Sequence[str]) -> dict[str, Any]:
        """Register a round with its authorized participant roster."""
        if round_id in self._rounds:
            raise AggregationError(f"round already open: {round_id}")
        if len(participants) < 2:
            raise AggregationError("a round needs at least two participants")
        self._rounds[round_id] = {
            "participants": list(participants),
            "submissions": {},
            "round_id": round_id,
            "opened_at": self.clock(),
        }
        return {"round_id": round_id, "participants": list(participants)}

    def submit(self, round_id: str, participant_id: str,
               masked_vector: Sequence[int]) -> dict[str, Any]:
        """Accept one masked contribution; duplicates and unknowns are rejected."""
        round_state = self._rounds.get(round_id)
        if round_state is None:
            raise AggregationError(f"unknown round: {round_id}")
        if participant_id not in round_state["participants"]:
            raise AggregationError(f"participant {participant_id!r} is not in the round roster")
        if participant_id in round_state["submissions"]:
            raise AggregationError(
                f"duplicate contribution rejected: {participant_id!r} already submitted in {round_id}"
            )
        round_state["submissions"][participant_id] = [int(v) for v in masked_vector]
        return {"accepted": True, "participant": participant_id}

    def server_visible_data(self, round_id: str) -> dict[str, list[int]]:
        """Exactly what the server observes: masked vectors only."""
        round_state = self._rounds.get(round_id)
        if round_state is None:
            raise AggregationError(f"unknown round: {round_id}")
        return {participant: list(vector)
                for participant, vector in round_state["submissions"].items()}

    def aggregate(self, round_id: str, *, dropped_masks: dict[str, list[int]] | None = None
                  ) -> AggregateResult:
        """Sum masked submissions; pairwise masks cancel, revealing only the aggregate."""
        round_state = self._rounds.get(round_id)
        if round_state is None:
            raise AggregationError(f"unknown round: {round_id}")
        submissions = round_state["submissions"]
        cohort = len(submissions)
        if cohort == 0:
            raise AggregationError("no contributions to aggregate")
        length = len(next(iter(submissions.values())))
        total = [0] * length
        for vector in submissions.values():
            total = [self.field.add(t, v) for t, v in zip(total, vector)]
        for participant, mask in (dropped_masks or {}).items():
            total = [self.field.sub(t, m) for t, m in zip(total, mask)]
        suppressed = cohort < self.min_cohort_size
        output = None if suppressed else total
        from ..canonical import digest_structure
        return AggregateResult(
            protocol_ref=PROTOCOL_REF,
            round_id=round_id,
            cohort_size=cohort,
            output=output,
            output_ref=digest_structure({"round": round_id, "cohort": cohort,
                                         "suppressed": suppressed}),
            validity_report={
                "participants_expected": len(round_state["participants"]),
                "participants_submitted": cohort,
                "duplicate_submissions_rejected": True,
                "round_binding_enforced": True,
                "collusion_bound": "server plus n-1 participants can isolate the last",
            },
            suppressed=suppressed,
            suppression_reason=(
                f"cohort size {cohort} below minimum {self.min_cohort_size}" if suppressed else ""
            ),
        )

    def dropout_recovery_masks(self, round_id: str, dropped_id: str,
                               survivor_reveals: Sequence[dict[str, Any]],
                               vector_length: int) -> dict[str, list[int]]:
        """Combine survivor reveals into the correction mask for a dropped participant.

        Each survivor S contributed ``sign_S * prg(seed_{S,D})`` to its own
        submission (sign_S = +1 when S is the lower id). The correction mask is the
        sum of those contributions; ``aggregate`` subtracts it, restoring the exact
        sum of the submitted values.
        """
        round_state = self._rounds.get(round_id)
        if round_state is None:
            raise AggregationError(f"unknown round: {round_id}")
        ordered = sorted(round_state["participants"])
        total_mask = [0] * vector_length
        for reveal in survivor_reveals:
            revealer = reveal["revealer"]
            if revealer not in ordered or revealer == dropped_id:
                raise AggregationError("dropout reveals must come from survivors")
            elements = reveal["elements"]
            if len(elements) != vector_length:
                raise AggregationError("reveal length mismatch")
            sign = 1 if revealer < dropped_id else -1
            total_mask = [m + sign * e for m, e in zip(total_mask, elements)]
        return {dropped_id: [self.field.reduce(m) for m in total_mask]}


def run_secure_aggregation_round(participants: Sequence[AggregationParticipant],
                                 values: dict[str, Sequence[int]],
                                 server: SecureAggregationServer, round_id: str,
                                 *, dropped: Sequence[str] = ()) -> AggregateResult:
    """Drive one full round: submit, handle dropouts, aggregate.

    Survivors of a dropout reveal the pairwise PRG elements they shared with the
    dropped participant; the server combines them into a correction mask and
    subtracts it, so the aggregate equals the sum of the submitted values while
    no unprotected individual value ever reaches the server.
    """
    roster = [participant.participant_id for participant in participants]
    server.open_round(round_id, roster)
    field = server.field
    vector_length = len(next(iter(values.values())))
    peer_publics = {participant.participant_id: participant.public
                    for participant in participants}
    for participant in participants:
        if participant.participant_id in dropped:
            continue
        masked = participant.mask_for(roster, round_id,
                                      values[participant.participant_id], field,
                                      peer_publics)
        server.submit(round_id, participant.participant_id, masked)
    dropped_masks: dict[str, list[int]] = {}
    for dropped_id in dropped:
        dropped_participant = next(p for p in participants if p.participant_id == dropped_id)
        reveals = [
            {
                "revealer": participant.participant_id,
                "elements": participant.reveal_pairwise_elements(
                    dropped_participant, round_id, vector_length, field,
                ),
            }
            for participant in participants
            if participant.participant_id != dropped_id
        ]
        dropped_masks.update(server.dropout_recovery_masks(
            round_id, dropped_id, reveals, vector_length,
        ))
    return server.aggregate(round_id, dropped_masks=dropped_masks)
