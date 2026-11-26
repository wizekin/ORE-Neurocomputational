"""Multi-party computation over additive shares with Beaver triples (step 47).

Parties hold private inputs and jointly compute ``f(x_1, ..., x_n)`` revealing only
the permitted outputs. Supported functions: secure vector sum across all parties and
secure inner product between two designated parties, both over the shared prime
field using trusted-dealer Beaver triples for multiplication.

Threat model (documented per spec steps 45, 47): semi-honest (honest-but-curious)
adversaries; input privacy holds against coalitions of up to n-1 parties because
additive shares are individually uniform; dropout is not tolerated by this protocol
family and aborts the session; the dealer sees triple values but never inputs.
"""
from __future__ import annotations

import time
from typing import Any, Callable, Sequence

from ..contracts import MPCSession
from ..exceptions import MPCError
from .fields import PrimeField
from .secret_sharing import AdditiveSecretSharer, BeaverTripleFactory

Clock = Callable[[], float]
PROTOCOL = "additive-sharing-beaver-v1"
THREAT_MODEL = (
    "semi-honest adversaries; input privacy against coalitions of up to n-1 "
    "parties; no dropout tolerance (session aborts); trusted-dealer triples"
)


class MPCProtocolEngine:
    """Execute secure sum and inner product on secret-shared inputs."""

    def __init__(self, *, field: PrimeField | None = None,
                 rng: Any = None, clock: Clock = time.time) -> None:
        self.field = field or PrimeField()
        self.clock = clock
        if rng is None:
            import numpy as np
            rng = np.random.default_rng()
        self.rng = rng
        self.sharer = AdditiveSecretSharer(self.field)
        self.dealer = BeaverTripleFactory(self.field)

    def create_session(self, session_id: str, participants: Sequence[str], *,
                       round_number: int = 1,
                       input_schema: str = "int-vector-61bit-field",
                       permitted_outputs: Sequence[str] = ("sum",),
                       collusion_threshold: int | None = None) -> MPCSession:
        """Instantiate a session with an explicit, documented threat model."""
        threshold = collusion_threshold
        if threshold is None:
            threshold = max(0, len(participants) - 1)
        return MPCSession(
            session_id=session_id,
            participants=tuple(participants),
            protocol=PROTOCOL,
            threat_model=THREAT_MODEL,
            collusion_threshold=threshold,
            round_number=round_number,
            input_schema=input_schema,
            permitted_outputs=tuple(permitted_outputs),
            dropout_policy="abort",
        )

    def secure_sum(self, session: MPCSession,
                   private_values: dict[str, Sequence[int]]) -> dict[str, Any]:
        """Compute the coordinate-wise sum of every party's private vector on shares.

        Each party's vector is additively shared among all participants; each party
        locally sums its own shares, and only the final per-party sums are combined
        to open the permitted aggregate. No party's plaintext vector is ever revealed.
        """
        self._validate(session, private_values, output="sum")
        lengths = {len(values) for values in private_values.values()}
        if len(lengths) > 1:
            raise MPCError("all parties must submit vectors of equal length")
        length = next(iter(lengths))
        if length == 0:
            raise MPCError("secure sum requires non-empty vectors")
        parties = list(session.participants)
        shared_vectors = [
            self.sharer.share_vector(values, len(parties), self.rng)
            for values in private_values.values()
        ]
        local_sums: list[list[int]] = []
        for party_index in range(len(parties)):
            total = [0] * length
            for shares in shared_vectors:
                total = [self.field.add(t, s)
                         for t, s in zip(total, shares[party_index].values)]
            local_sums.append(total)
        output = [
            sum(local_sums[party_index][index] for party_index in range(len(parties)))
            % self.field.modulus
            for index in range(length)
        ]
        return {
            "session_id": session.session_id,
            "output": output,
            "output_kind": "sum",
            "revealed": ["sum"],
            "parties_sharing": len(parties),
            "note": "only the permitted aggregate output is revealed",
        }

    def secure_inner_product(self, session: MPCSession,
                             party_a: str, vector_a: Sequence[int],
                             party_b: str, vector_b: Sequence[int]) -> dict[str, Any]:
        """Compute the inner product of two parties' private vectors via Beaver triples.

        The parties never reveal their vectors: each coordinate product is computed
        on additive shares with a Beaver triple, and only the final sum is opened.
        """
        if len(vector_a) != len(vector_b):
            raise MPCError("inner product requires vectors of equal length")
        if party_a == party_b:
            raise MPCError("inner product requires two distinct parties")
        self._validate(session, {party_a: vector_a, party_b: vector_b}, output="inner_product")
        length = len(vector_a)
        if length == 0:
            raise MPCError("inner product requires non-empty vectors")
        parties = list(session.participants)
        # share both vectors among all session parties
        shares_a = self.sharer.share_vector(vector_a, len(parties), self.rng)
        shares_b = self.sharer.share_vector(vector_b, len(parties), self.rng)
        triples = self.dealer.generate(len(parties), length, self.rng)
        running = [0] * len(parties)  # additive shares of the running product sum
        for index in range(length):
            a_shares, b_shares, c_shares = BeaverTripleFactory.triple_at(triples, index)
            x_shares = [shares_a[p].values[index] for p in range(len(parties))]
            y_shares = [shares_b[p].values[index] for p in range(len(parties))]
            # open d = x - a and e = y - b (masked by the random triple)
            d = sum(x_shares) - sum(a_shares)
            e = sum(y_shares) - sum(b_shares)
            # z_i = c_i + e*a_i + d*b_i ; party 0 additionally adds d*e
            for party in range(len(parties)):
                term = (c_shares[party]
                        + e * a_shares[party]
                        + d * b_shares[party])
                if party == 0:
                    term += d * e
                running[party] = self.field.add(running[party], term)
        opened = sum(running) % self.field.modulus
        return {
            "session_id": session.session_id,
            "output": opened,
            "output_kind": "inner_product",
            "revealed": ["inner_product"],
            "beaver_triples_used": length,
            "note": "vectors remained secret-shared; only the product sum was opened",
        }

    def _validate(self, session: MPCSession, provided: dict[str, Sequence[int]],
                  *, output: str) -> None:
        if output not in session.permitted_outputs:
            raise MPCError(
                f"output {output!r} is not in the session's permitted outputs"
            )
        for party in provided:
            if party not in session.participants:
                raise MPCError(f"party {party!r} is not a session participant")
        for values in provided.values():
            for value in values:
                if not isinstance(value, int) or value < 0 or value >= self.field.modulus:
                    raise MPCError(
                        "inputs must be non-negative integers below the field modulus"
                    )
