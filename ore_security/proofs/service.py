"""Proof service: task binding, independent verification, result records (steps 29-31).

The service binds each proof to its statement (task id, nonce, expiry, verifier
context), verifies independently of the prover, and returns a ``ProofVerificationResult``
recording the statement and verifier versions. A failed verification never marks an
operation as verified.
"""
from __future__ import annotations

import time
import uuid
from typing import Any, Callable

from ..contracts import ProofStatement, ProofVerificationResult
from ..exceptions import ProofError, StatementVersionError
from .schnorr import (
    ADAPTER_ID,
    RELATION_ASSUMPTIONS,
    RELATION_DESCRIPTION,
    RELATION_ID,
    RELATION_VERSION,
    SchnorrGroup,
    SchnorrProof,
    SchnorrProofAdapter,
    SchnorrWitness,
)
from .statements import ProofStatementRegistry, RelationDefinition

Clock = Callable[[], float]
VERIFIER_VERSION = "ore-security-proof-verifier-1.0.0"


class ProofService:
    """Statement registry plus reference Schnorr prover and independent verifier."""

    def __init__(self, *, clock: Clock = time.time,
                 registry: ProofStatementRegistry | None = None,
                 adapter: SchnorrProofAdapter | None = None) -> None:
        self.clock = clock
        self.registry = registry or ProofStatementRegistry()
        self.adapter = adapter or SchnorrProofAdapter()
        self._register_default_relation()
        self._records: dict[str, dict[str, Any]] = {}

    def _register_default_relation(self) -> None:
        """Register the reference discrete-log relation with its schema."""
        self.registry.register_relation(RelationDefinition(
            relation_id=RELATION_ID,
            version=RELATION_VERSION,
            description=RELATION_DESCRIPTION,
            public_input_schema={"y": "int"},
            assumptions=RELATION_ASSUMPTIONS,
            motivation=(
                "narrow claim used for credential predicates: a participant proves "
                "knowledge of a committed secret without revealing it"
            ),
        ))

    def register_relation(self, definition: RelationDefinition, *,
                          replace: bool = False) -> None:
        """Register an additional relation version (upgrade control, step 31)."""
        self.registry.register_relation(definition, replace=replace)

    def build_statement(self, statement_id: str, public_inputs: dict[str, Any], *,
                        task_id: str, task_nonce: str, ttl: float = 600.0,
                        relation_id: str = RELATION_ID,
                        relation_version: str = RELATION_VERSION) -> ProofStatement:
        """Create a task-bound statement for the reference relation."""
        return self.registry.build_statement(
            statement_id, relation_id, relation_version, public_inputs,
            task_id=task_id, task_nonce=task_nonce,
            expires_at=self.clock() + float(ttl),
        )

    def prove(self, statement: ProofStatement, witness: SchnorrWitness) -> dict[str, Any]:
        """Produce a proof bound to the statement's task context."""
        proof = self.adapter.prove(witness, statement.binding_payload(),
                                   dict(statement.public_inputs))
        record = {
            "proof_id": "proof-" + uuid.uuid4().hex[:16],
            "statement_id": statement.statement_id,
            "relation_id": statement.relation_id,
            "relation_version": statement.relation_version,
            "adapter": ADAPTER_ID,
            "proof": proof.to_dict(),
            "created_at": self.clock(),
        }
        self._records[record["proof_id"]] = record
        return record

    def verify(self, proof_record: dict[str, Any],
               statement: ProofStatement) -> ProofVerificationResult:
        """Independently verify a proof record against a statement.

        Checks statement binding (task id, nonce, expiry, verifier context), relation
        version support, and the cryptographic verification itself. Any failure
        returns ``verified=False`` with a recorded reason; verifier and statement
        versions are always recorded.
        """
        proof_id = proof_record.get("proof_id", "")
        now = self.clock()

        def result(verified: bool, reason: str) -> ProofVerificationResult:
            return ProofVerificationResult(
                proof_ref=proof_id or "unknown",
                statement_id=statement.statement_id,
                relation_id=statement.relation_id,
                relation_version=statement.relation_version,
                verifier_version=VERIFIER_VERSION,
                verified=verified,
                reason=reason,
                checked_at=now,
            )

        if proof_record.get("statement_id") != statement.statement_id:
            return result(False, "proof is bound to a different statement")
        if proof_record.get("relation_version") != statement.relation_version:
            return result(False, "proof relation version differs from the statement")
        if now >= statement.expires_at:
            return result(False, "statement expired")
        try:
            definition = self.registry.get_relation(statement.relation_id,
                                                    statement.relation_version)
            if definition is None or not definition.supported:
                return result(False, "relation version is retired or unknown")
            group = SchnorrGroup(
                int(proof_record["proof"]["group"]["p"]),
                int(proof_record["proof"]["group"]["q"]),
                int(proof_record["proof"]["group"]["g"]),
            )
            proof = SchnorrProof(
                t=int(proof_record["proof"]["t"]),
                c=int(proof_record["proof"]["c"]),
                s=int(proof_record["proof"]["s"]),
                group=group,
                binding_digest=proof_record["proof"]["binding_digest"],
            )
            if not self.adapter.verify(proof, statement.binding_payload(),
                                       dict(statement.public_inputs)):
                return result(False, "cryptographic verification failed")
        except (KeyError, TypeError, ValueError, ProofError, StatementVersionError) as exc:
            return result(False, f"malformed proof record: {exc}")
        return result(True, "statement binding and cryptographic verification passed")

    def record(self, proof_id: str) -> dict[str, Any] | None:
        """Retrieve a stored proof record (metadata and proof object)."""
        return self._records.get(proof_id)
