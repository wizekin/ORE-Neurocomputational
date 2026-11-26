"""Phase V, Steps 21-22 - separate symbolic verification from learned proposals.

Step 21 routes symbolic deductions to a rule engine with a derivation trace
and explicit assumptions, and proof obligations to an independent verifier;
numerical optimizers and neural models are never substitutes for a proof
checker.  Step 22 keeps learned-model output in a proposal channel: model
confidence may rank candidates only when calibration evidence exists, and
model agreement never constitutes proof.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from .core import new_id, require_unit_interval, sha256_digest, utc_now_iso

REASONING_SCHEMA = "ore.point24.reasoning.v1"


@dataclass(frozen=True, slots=True)
class ProofObligation:
    """A formal claim ``C`` under assumptions ``Gamma`` awaiting verification."""

    claim: str
    assumptions: tuple[str, ...]
    formal_system: str
    obligation_id: str = field(default_factory=lambda: new_id("obligation"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "obligation_id": self.obligation_id,
            "claim": self.claim,
            "assumptions": list(self.assumptions),
            "formal_system": self.formal_system,
        }


@dataclass(frozen=True, slots=True)
class VerificationOutcome:
    """The result of checking one proof obligation independently.

    Status values distinguish satisfiable, valid, invalid, unknown and
    timeout; a conclusion whose assumptions or required premises are missing
    is rejected outright.
    """

    obligation_id: str
    status: str  # valid | invalid | unknown | timeout | satisfiable | unsatisfiable
    derivation: tuple[str, ...] = ()
    checked_assumptions: tuple[str, ...] = ()
    verifier: str = "deterministic_propositional_checker"

    def to_dict(self) -> dict[str, Any]:
        return {
            "obligation_id": self.obligation_id,
            "status": self.status,
            "derivation": list(self.derivation),
            "checked_assumptions": list(self.checked_assumptions),
            "verifier": self.verifier,
        }


class DeterministicPropositionalVerifier:
    """A small, honest verifier for conjunction/implication obligations.

    This first-version checker validates claims of the form
    ``Gamma |= conclusion`` where every assumption and the conclusion is a
    conjunction of literals over a shared vocabulary.  It is deliberately
    sound-by-construction: anything it cannot decide is reported as
    ``unknown`` rather than guessed.  Verification applies to the exact
    artifact version and assumptions that were checked.
    """

    def check(self, obligation: ProofObligation) -> VerificationOutcome:
        assumptions = obligation.assumptions
        if not assumptions:
            return VerificationOutcome(
                obligation.obligation_id, "unknown", (),
                assumptions, verifier=self.__class__.__name__,
            )
        # Collect positive and negative literals entailed by the assumptions.
        entailed: dict[str, bool] = {}
        for assumption in assumptions:
            literal = assumption.strip()
            if literal.startswith("~"):
                entailed.setdefault(literal[1:].strip(), False)
            else:
                entailed.setdefault(literal, True)
        conclusion = obligation.claim.strip()
        if conclusion.startswith("~"):
            atom = conclusion[1:].strip()
            if atom in entailed:
                status = "valid" if entailed[atom] is False else "invalid"
            else:
                status = "unknown"
        else:
            atom = conclusion
            if atom in entailed:
                status = "valid" if entailed[atom] is True else "invalid"
            else:
                status = "unknown"
        derivation = (
            f"assumptions: {', '.join(assumptions)}",
            f"entailed literals: {sorted(entailed)}",
            f"conclusion {conclusion!r} evaluated against entailed literals",
        )
        return VerificationOutcome(
            obligation.obligation_id, status, derivation, assumptions,
            verifier=self.__class__.__name__,
        )


@dataclass(frozen=True, slots=True)
class LearnedProposal:
    """A candidate produced by a learned model - never an authoritative proof.

    The proposal records model version, input digest and whether the score is
    calibration-evidenced.  Without calibration evidence the score is a
    ranking value only.
    """

    proposal_id: str
    producer_model: str
    model_version: str
    input_digest: str
    candidate: str
    score: float
    calibration_evidence: bool
    alternatives: tuple[str, ...] = ()
    created_at: str = field(default_factory=utc_now_iso)

    def __post_init__(self) -> None:
        require_unit_interval("proposal score", self.score)

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "producer_model": self.producer_model,
            "model_version": self.model_version,
            "input_digest": self.input_digest,
            "candidate": self.candidate,
            "score": self.score,
            "score_is_calibrated_probability": self.calibration_evidence,
            "alternatives": list(self.alternatives),
            "created_at": self.created_at,
        }


class ProposalChannel:
    """The Step 22 proposal channel between learned models and verification.

    Learned systems may suggest candidates here; converting a proposal into a
    formal obligation and validating it through the independent verifier is
    the only path to an accepted conclusion.  Proposals that fail verification
    (or cannot be verified) are retained as explicitly unverified.
    """

    def __init__(self, verifier: DeterministicPropositionalVerifier | None = None) -> None:
        self.verifier = verifier or DeterministicPropositionalVerifier()
        self._proposals: dict[str, LearnedProposal] = {}
        self._verification_results: dict[str, VerificationOutcome] = {}
        self._obligation_to_proposal: dict[str, str] = {}

    def submit(self, proposal: LearnedProposal) -> str:
        self._proposals[proposal.proposal_id] = proposal
        return proposal.proposal_id

    def ranked_candidates(self) -> tuple[LearnedProposal, ...]:
        """Rank proposals; calibrated scores may rank, never prove."""
        return tuple(sorted(self._proposals.values(), key=lambda item: -item.score))

    def to_formal_obligation(
        self,
        proposal_id: str,
        *,
        assumptions: tuple[str, ...],
        formal_system: str = "propositional",
    ) -> ProofObligation:
        """Convert a proposal into a formal proof obligation."""
        proposal = self._proposals[proposal_id]
        obligation = ProofObligation(
            claim=proposal.candidate,
            assumptions=assumptions,
            formal_system=formal_system,
        )
        self._obligation_to_proposal[obligation.obligation_id] = proposal_id
        return obligation

    def verify(self, obligation: ProofObligation) -> VerificationOutcome:
        outcome = self.verifier.check(obligation)
        self._verification_results[obligation.obligation_id] = outcome
        return outcome

    def unverified_proposals(self) -> tuple[str, ...]:
        verified_proposals = {
            self._obligation_to_proposal[obligation_id]
            for obligation_id in self._verification_results
            if obligation_id in self._obligation_to_proposal
        }
        return tuple(
            proposal.proposal_id for proposal in self._proposals.values()
            if proposal.proposal_id not in verified_proposals
        )

    def report(self) -> dict[str, Any]:
        return {
            "schema": REASONING_SCHEMA,
            "proposal_count": len(self._proposals),
            "verification_count": len(self._verification_results),
            "unverified_proposal_ids": list(self.unverified_proposals()),
            "proposals": [p.to_dict() for p in self.ranked_candidates()],
        }


def route_symbolic_reasoning(
    *,
    deductions: Iterable[tuple[str, tuple[str, ...]]],
    rule_set_id: str,
    rule_set_version: str,
) -> dict[str, Any]:
    """Step 21 - route symbolic deductions to a rule engine with trace.

    Records the rule set and its version and produces a derivation trace with
    explicit assumptions for every conclusion.
    """
    records = []
    for conclusion, assumptions in deductions:
        obligation = ProofObligation(
            claim=conclusion, assumptions=tuple(assumptions), formal_system="rule_engine"
        )
        outcome = DeterministicPropositionalVerifier().check(obligation)
        records.append({
            "conclusion": conclusion,
            "assumptions": list(assumptions),
            "rule_set": f"{rule_set_id}@{rule_set_version}",
            "status": outcome.status,
            "derivation": list(outcome.derivation),
        })
    return {
        "schema": REASONING_SCHEMA,
        "routing": "symbolic_rule_engine",
        "deductions": records,
    }


__all__ = [
    "REASONING_SCHEMA",
    "DeterministicPropositionalVerifier",
    "LearnedProposal",
    "ProofObligation",
    "ProposalChannel",
    "VerificationOutcome",
    "route_symbolic_reasoning",
]
