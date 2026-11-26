"""VDAF/Prio-style aggregation with private validity checks (steps 49-51).

Clients secret-share binary measurement vectors between two non-colluding
aggregators. Validity uses the classic Prio affine AGREE check: for a binary
vector x, every coordinate satisfies x_i^2 = x_i, so

    u = sum_i (x_i^2 - x_i) = 0 (mod p)

The servers evaluate u on shares with a Beaver triple, opening only the masked
intermediates and the final u, never the measurement. Valid contributions are
aggregated per coordinate; cohorts below the configured minimum are suppressed.

Documented limitation: the affine AGGREE check over the full field binds the
relation sum(x_i^2 - x_i) = 0, which honest binary measurements satisfy. A fully
malicious client can in principle craft non-binary field vectors that satisfy the
same relation; production Prio adds structured-ring checks per coordinate. The
protocol reference and cohort metadata are recorded with every aggregate.
"""
from __future__ import annotations

import time
from typing import Any, Callable, Sequence

from ..contracts import AggregateResult
from ..exceptions import VDAFError
from ..multiparty.fields import PrimeField
from ..multiparty.secret_sharing import AdditiveSecretSharer, BeaverTripleFactory

Clock = Callable[[], float]
PROTOCOL_REF = "prio-affine-agree-v1"


class PrioClient:
    """Encode one binary measurement as secret shares for the two aggregators."""

    def __init__(self, field: PrimeField | None = None, rng: Any = None) -> None:
        self.field = field or PrimeField()
        if rng is None:
            import numpy as np
            rng = np.random.default_rng()
        self.rng = rng
        self.sharer = AdditiveSecretSharer(self.field)

    def encode(self, measurement: Sequence[int]) -> dict[str, Any]:
        """Validate the measurement structurally and split it into two shares."""
        if not measurement:
            raise VDAFError("measurement must be a non-empty vector")
        for value in measurement:
            if not isinstance(value, int) or value < 0 or value > 1:
                raise VDAFError(
                    f"measurement values must be binary 0/1 field elements; got {value!r}"
                )
        share_a = [self.field.random(self.rng) for _ in measurement]
        share_b = [self.field.sub(value, share)
                   for value, share in zip(measurement, share_a)]
        return {
            "protocol": PROTOCOL_REF,
            "length": len(measurement),
            "share_a": share_a,
            "share_b": share_b,
        }


class PrioAggregationRound:
    """One aggregation round run by the two non-colluding aggregator servers."""

    def __init__(self, round_id: str, *, field: PrimeField | None = None,
                 min_cohort_size: int = 3, rng: Any = None,
                 clock: Clock = time.time) -> None:
        self.round_id = round_id
        self.field = field or PrimeField()
        self.min_cohort_size = int(min_cohort_size)
        self.clock = clock
        if rng is None:
            import numpy as np
            rng = np.random.default_rng()
        self.rng = rng
        self.sharer = AdditiveSecretSharer(self.field)
        self.dealer = BeaverTripleFactory(self.field)
        self._accepted: list[dict[str, list[int]]] = []
        self._rejected: list[dict[str, Any]] = []
        self._participants: set[str] = set()

    def submit(self, client_id: str, encoded: dict[str, Any]) -> dict[str, Any]:
        """Accept one encoded measurement after the private validity check."""
        if not isinstance(encoded, dict) or encoded.get("protocol") != PROTOCOL_REF:
            raise VDAFError("measurement is not an encoded Prio submission")
        share_a = encoded.get("share_a")
        share_b = encoded.get("share_b")
        length = encoded.get("length")
        if (not isinstance(share_a, list) or not isinstance(share_b, list)
                or length != len(share_a) or length != len(share_b) or not length):
            raise VDAFError("malformed measurement shares: lengths disagree")
        for share in share_a + share_b:
            if not isinstance(share, int) or not 0 <= share < self.field.modulus:
                raise VDAFError("malformed measurement shares: values outside the field")
        if client_id in self._participants:
            raise VDAFError(
                f"duplicate participant contribution rejected in round {self.round_id}"
            )
        self._participants.add(client_id)
        validity = self._check_validity(share_a, share_b)
        if validity["valid"]:
            self._accepted.append({"client_id": client_id,
                                   "share_a": share_a, "share_b": share_b})
        else:
            self._rejected.append({"client_id": client_id,
                                   "reason": validity["reason"]})
        return validity

    def _check_validity(self, share_a: list[int], share_b: list[int]) -> dict[str, Any]:
        """Affine AGREE validity: u = sum(x_i^2 - x_i) must be zero (mod p)."""
        length = len(share_a)
        triple = self.dealer.generate(2, length, self.rng)
        running_a = 0  # server A's share of u
        running_b = 0
        for index in range(length):
            a_shares, b_shares, c_shares = BeaverTripleFactory.triple_at(triple, index)
            x_a, x_b = share_a[index], share_b[index]
            # compute z = x*x on shares with the Beaver triple
            d = (x_a + x_b - (a_shares[0] + a_shares[1])) % self.field.modulus
            e = (x_a + x_b - (b_shares[0] + b_shares[1])) % self.field.modulus
            z_a = (c_shares[0] + e * a_shares[0] + d * b_shares[0]) % self.field.modulus
            z_b = (c_shares[1] + e * a_shares[1] + d * b_shares[1] + d * e) % self.field.modulus
            running_a = (running_a + z_a - x_a) % self.field.modulus
            running_b = (running_b + z_b - x_b) % self.field.modulus
        u = (running_a + running_b) % self.field.modulus
        return {
            "valid": u == 0,
            "opened_check_value": u,
            "reason": "affine AGREE relation sum(x_i^2 - x_i) = 0 mod p",
        }

    def aggregate(self) -> AggregateResult:
        """Sum valid shares per coordinate; suppress small cohorts (step 50)."""
        cohort = len(self._accepted)
        suppressed = cohort < self.min_cohort_size
        if self._accepted:
            length = len(self._accepted[0]["share_a"])
            totals_a = [0] * length
            totals_b = [0] * length
            for submission in self._accepted:
                totals_a = [self.field.add(t, s)
                            for t, s in zip(totals_a, submission["share_a"])]
                totals_b = [self.field.add(t, s)
                            for t, s in zip(totals_b, submission["share_b"])]
            output = [self.field.add(a, b)
                      for a, b in zip(totals_a, totals_b)] if not suppressed else None
        else:
            output = None
        from ..canonical import digest_structure
        return AggregateResult(
            protocol_ref=PROTOCOL_REF,
            round_id=self.round_id,
            cohort_size=cohort,
            output=output,
            output_ref=digest_structure({
                "round": self.round_id, "cohort": cohort,
                "rejected": len(self._rejected), "suppressed": suppressed,
            }),
            validity_report={
                "valid_contributions": cohort,
                "invalid_contributions": len(self._rejected),
                "rejected_reasons": [item["reason"] for item in self._rejected][:10],
                "duplicate_participants_rejected": True,
                "protocol_version": PROTOCOL_REF,
                "validity_relation": "sum(x_i^2 - x_i) = 0 mod p on secret shares",
                "limitation": "affine AGREE over the full field; structured-ring "
                              "per-coordinate checks are required for malicious clients",
            },
            suppressed=suppressed,
            suppression_reason=(
                f"cohort size {cohort} below minimum {self.min_cohort_size}"
                if suppressed else ""
            ),
        )
