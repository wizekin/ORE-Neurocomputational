"""Reference Schnorr proof-of-knowledge adapter (steps 27-28).

Relation (discrete-logarithm knowledge):

    R(x, w) = 1  iff  g^w = y (mod p)

The prover demonstrates knowledge of the witness w without revealing it, using a
Fiat-Shamir-transformed Schnorr sigma protocol over a safe-prime subgroup.

Security properties, stated explicitly per spec step 28:

* Completeness: an honest prover convinces the verifier with probability 1.
* Special soundness / knowledge soundness: under the discrete-logarithm assumption
  in the subgroup, a prover that answers two distinct challenges for the same
  commitment reveals w; the Fiat-Shamir challenge is derived from a SHA-256 random
  oracle over the full transcript including the task binding.
* Honest-verifier zero knowledge: the transcript (t, c, s) is simulatable; the
  adapter targets HVZK, not simulation against arbitrary malicious verifiers.
* Setup: no trusted setup; the group is a generated safe-prime subgroup.
* Assumptions: DL hardness in the subgroup, random-oracle model for SHA-256.

This is a reference adapter behind the typed proof contracts. Production systems
must bind a maintained proof-system library through the same adapter interface; the
default group below was generated once with the documented deterministic procedure.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..canonical import canonical_bytes, digest_bytes
from ..crypto import random_int
from ..exceptions import ProofError

# Generated with scripts/gen_schnorr_group.py: q prime, p = 2q+1 prime, g of order q.
DEFAULT_P = 97513438229728082416093130183230488297568635423984915056744967651937392000419
DEFAULT_Q = 48756719114864041208046565091615244148784317711992457528372483825968696000209
DEFAULT_G = 4

ADAPTER_ID = "schnorr-dlog-reference-1.0.0"
RELATION_ID = "dlog-knowledge"
RELATION_VERSION = "1.0.0"
RELATION_DESCRIPTION = "knowledge of w such that g^w = y mod p in a safe-prime subgroup"
RELATION_ASSUMPTIONS = (
    "discrete-logarithm hardness in the order-q subgroup; SHA-256 random oracle; "
    "honest-verifier zero knowledge; no trusted setup"
)


@dataclass(frozen=True, slots=True)
class SchnorrGroup:
    """Public group parameters for the reference adapter."""

    p: int
    q: int
    g: int

    def __post_init__(self) -> None:
        if self.p != 2 * self.q + 1:
            raise ProofError("group must satisfy p = 2q + 1 (safe-prime structure)")
        if not 1 < self.g < self.p:
            raise ProofError("generator out of range")

    def to_dict(self) -> dict[str, str]:
        return {"p": str(self.p), "q": str(self.q), "g": str(self.g)}


@dataclass(frozen=True, slots=True)
class SchnorrWitness:
    """Private witness w; never serialized into statements, proofs or logs."""

    w: int

    def __post_init__(self) -> None:
        if self.w <= 0:
            raise ProofError("witness must be a positive integer")

    def public_value(self, group: SchnorrGroup) -> int:
        """Public statement input y = g^w mod p."""
        return pow(group.g, self.w, group.p)


@dataclass(frozen=True, slots=True)
class SchnorrProof:
    """Non-interactive proof: commitment t, challenge c, response s, binding."""

    t: int
    c: int
    s: int
    group: SchnorrGroup
    binding_digest: str

    def to_dict(self) -> dict[str, str]:
        return {
            "adapter": ADAPTER_ID,
            "t": str(self.t),
            "c": str(self.c),
            "s": str(self.s),
            "group": self.group.to_dict(),
            "binding_digest": self.binding_digest,
        }


class SchnorrProofAdapter:
    """Prover/verifier for the discrete-log relation with task binding."""

    def __init__(self, group: SchnorrGroup | None = None) -> None:
        self.group = group or SchnorrGroup(DEFAULT_P, DEFAULT_Q, DEFAULT_G)

    def prove(self, witness: SchnorrWitness, binding_payload: dict,
              public_inputs: dict) -> SchnorrProof:
        """Prove knowledge of w for y = g^w mod p, bound to the task context."""
        y = public_inputs.get("y")
        if not isinstance(y, int) or y != witness.public_value(self.group):
            raise ProofError("public input y does not match the witness")
        binding_digest = digest_bytes(canonical_bytes(binding_payload))
        nonce = random_int(self.group.q)
        t = pow(self.group.g, nonce, self.group.p)
        c = self._challenge(binding_digest, t, public_inputs)
        s = (nonce + c * witness.w) % self.group.q
        return SchnorrProof(t=t, c=c, s=s, group=self.group, binding_digest=binding_digest)

    def verify(self, proof: SchnorrProof, binding_payload: dict,
               public_inputs: dict) -> bool:
        """Verify the proof against the recomputed challenge and group equation."""
        y = public_inputs.get("y")
        if not isinstance(y, int) or not 0 < y < proof.group.p:
            return False
        if proof.t <= 0 or proof.t >= proof.group.p:
            return False
        if not 0 <= proof.c < proof.group.q or not 0 <= proof.s < proof.group.q:
            return False
        expected_binding = digest_bytes(canonical_bytes(binding_payload))
        if proof.binding_digest != expected_binding:
            return False
        if self._challenge(proof.binding_digest, proof.t, public_inputs) != proof.c:
            return False
        left = pow(proof.group.g, proof.s, proof.group.p)
        right = (proof.t * pow(y, proof.c, proof.group.p)) % proof.group.p
        return left == right

    def _challenge(self, binding_digest: str, commitment: int,
                   public_inputs: dict) -> int:
        """Fiat-Shamir challenge over binding, commitment and public inputs."""
        transcript = {
            "adapter": ADAPTER_ID,
            "relation": RELATION_ID,
            "binding_digest": binding_digest,
            "commitment": str(commitment),
            "public_inputs": {key: str(value) for key, value in sorted(public_inputs.items())},
        }
        return int.from_bytes(bytes.fromhex(digest_bytes(canonical_bytes(transcript))), "big") % self.group.q
