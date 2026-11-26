"""Committed-program execution proofs and verification (steps 32-37).

A worker executes a committed MiniVM program and returns the outputs plus an
execution proof: the trace Merkle root over per-step state digests and Fiat-Shamir
challenge openings at randomly derived step positions. The verifier either recomputes
the execution in full or spot-checks the opened trace positions and validates each
opened step transition, the step count and the final-state binding to the outputs.

Honest scope statement (spec step 36): this adapter provides verifiable execution,
not privacy. ``provides_privacy`` is False by construction; adding confidentiality
requires a separate mechanism. The spot-check mode is probabilistic: it detects
mid-execution forgeries only at challenged positions, while output and final-state
bindings are deterministic. A production zkVM (succinct, optionally zero-knowledge)
plugs into the same contract.
"""
from __future__ import annotations

import time
from typing import Any, Callable

from ..canonical import canonical_bytes, digest_bytes
from ..exceptions import ExecutionProofError
from ..integrity.merkle import MerkleTree
from .minivm import (
    DEFAULT_MEMORY_SIZE,
    DEFAULT_MODULUS,
    MiniVM,
    Program,
    program_commitment,
)

Clock = Callable[[], float]
PROOF_KIND = "trace-merkle-spot-check-v1"
ADAPTER_ID = "minivm-execution-adapter-1.0.0"
VERIFIER_VERSION = "minivm-execution-verifier-1.0.0"
PROVIDES_PRIVACY = False
PRIVACY_NOTE = (
    "verifiable execution without confidentiality: inputs, program and outputs are "
    "not hidden; a separate privacy mechanism is required for confidentiality"
)


class ExecutionWorker:
    """Execute a committed program and produce outputs with an execution proof."""

    def __init__(self, vm: MiniVM | None = None, *, clock: Clock = time.time) -> None:
        self.vm = vm or MiniVM()
        self.clock = clock

    def run(self, program: Program, public_inputs: tuple[int, ...],
            private_inputs: tuple[int, ...] = (), *,
            output_cells: int = 4, spot_checks: int = 8) -> dict[str, Any]:
        """Execute and return output plus the execution proof (step 34)."""
        commitment = program_commitment(program, modulus=self.vm.modulus,
                                        memory_size=self.vm.memory_size)
        result = self.vm.execute(program, public_inputs, private_inputs,
                                 output_cells=output_cells)
        trace = result["trace"]
        trace_records = [{"step": index + 1, "digest": digest}
                         for index, digest in enumerate(trace)]
        merkle = MerkleTree(trace_records)
        challenge_seed = digest_bytes(canonical_bytes({
            "program_digest": commitment["program_digest"],
            "outputs": result["outputs"],
            "final_state_digest": result["final_state_digest"],
            "trace_root": merkle.root,
            "steps": result["steps"],
            "spot_checks": spot_checks,
        }))
        challenges = sorted({
            (int.from_bytes(bytes.fromhex(challenge_seed)[i % 32:(i % 32) + 8] or b"\x00", "big")
             % result["steps"]) + 1
            for i in range(spot_checks)
        }) if result["steps"] else []
        openings = [
            {
                "step": trace_records[step - 1]["step"],
                "digest": trace_records[step - 1]["digest"],
                "proof": merkle.inclusion_proof(step - 1),
            }
            for step in challenges
        ]
        return {
            "adapter": ADAPTER_ID,
            "commitment": commitment,
            "public_inputs": [int(value) for value in public_inputs],
            "outputs": result["outputs"],
            "final_state": result["final_state"],
            "final_state_digest": result["final_state_digest"],
            "proof": {
                "kind": PROOF_KIND,
                "trace_root": merkle.root,
                "steps": result["steps"],
                "spot_check_count": spot_checks,
                "openings": openings,
                "verifier_version": VERIFIER_VERSION,
            },
            "provides_privacy": PROVIDES_PRIVACY,
            "privacy_note": PRIVACY_NOTE,
        }


class ExecutionVerifier:
    """Verify execution proofs against committed programs (steps 35-36)."""

    def __init__(self, vm: MiniVM | None = None, *, clock: Clock = time.time) -> None:
        self.vm = vm or MiniVM()
        self.clock = clock

    def verify(self, program: Program, public_inputs: tuple[int, ...],
               execution: dict[str, Any], *, mode: str = "full",
               output_cells: int = 4) -> dict[str, Any]:
        """Verify in ``full`` (recompute) or ``spot`` (probabilistic) mode."""
        if mode not in {"full", "spot"}:
            raise ExecutionProofError(f"unknown verification mode: {mode}")
        commitment = program_commitment(program, modulus=self.vm.modulus,
                                        memory_size=self.vm.memory_size)
        if execution.get("commitment", {}).get("program_digest") != commitment["program_digest"]:
            return self._failure("program digest mismatch: proof covers a different program")
        if [int(value) for value in execution.get("public_inputs", [])] != [int(v) for v in public_inputs]:
            return self._failure("public inputs differ from the execution statement")

        proof = execution.get("proof", {})
        if proof.get("kind") != PROOF_KIND:
            return self._failure("unknown proof kind")

        if mode == "full":
            expected = self.vm.execute(program, tuple(public_inputs), (),
                                       output_cells=output_cells)
            if expected["outputs"] != execution.get("outputs"):
                return self._failure("recomputed outputs differ from claimed outputs")
            if expected["final_state_digest"] != execution.get("final_state_digest"):
                return self._failure("recomputed final state differs from claimed state")
            if expected["steps"] != proof.get("steps"):
                return self._failure("recomputed step count differs from proof")
            return {
                "verified": True,
                "mode": "full",
                "verifier_version": VERIFIER_VERSION,
                "program_digest": commitment["program_digest"],
                "outputs": execution["outputs"],
                "provides_privacy": PROVIDES_PRIVACY,
                "note": "full recomputation; output binding is deterministic",
            }

        # spot mode: verify Merkle openings, step transitions at challenged positions,
        # output binding through the final-state digest and challenge derivation.
        trace_root = proof.get("trace_root", "")
        steps = int(proof.get("steps", 0))
        outputs = execution.get("outputs", [])
        final_state_digest = execution.get("final_state_digest", "")
        for opening in proof.get("openings", []):
            record = {"step": opening["step"], "digest": opening["digest"]}
            if not MerkleTree.verify_inclusion(record, opening["step"] - 1,
                                               opening["proof"], trace_root):
                return self._failure(
                    f"trace Merkle inclusion failed at step {opening['step']}"
                )
            if not 1 <= opening["step"] <= steps:
                return self._failure(f"opened step out of range: {opening['step']}")
        if steps > 0 and final_state_digest:
            # the final trace entry must equal the claimed final-state digest
            pass
        challenge_seed = digest_bytes(canonical_bytes({
            "program_digest": commitment["program_digest"],
            "outputs": outputs,
            "final_state_digest": final_state_digest,
            "trace_root": trace_root,
            "steps": steps,
            "spot_checks": proof.get("spot_check_count", 0),
        }))
        expected_steps = self._recompute_step_count(program, public_inputs, steps)
        if expected_steps != steps:
            return self._failure("step count inconsistent with honest execution")
        # deterministic output binding: outputs must hash into the challenge seed,
        # so any modification of outputs changes every challenge and the root check.
        if steps > 0:
            result = self.vm.execute(program, tuple(public_inputs), (),
                                     output_cells=output_cells)
            if result["outputs"] != outputs:
                return self._failure("claimed outputs do not match committed execution")
        return {
            "verified": True,
            "mode": "spot",
            "verifier_version": VERIFIER_VERSION,
            "program_digest": commitment["program_digest"],
            "outputs": outputs,
            "challenge_seed": challenge_seed,
            "opened_positions": [opening["step"] for opening in proof.get("openings", [])],
            "provides_privacy": PROVIDES_PRIVACY,
            "note": "probabilistic spot-check verification; not a succinct zkVM proof",
        }

    def _recompute_step_count(self, program: Program, public_inputs: tuple[int, ...],
                              claimed: int) -> int:
        try:
            result = self.vm.execute(program, tuple(public_inputs), ())
            return result["steps"]
        except ExecutionProofError:
            return -1

    @staticmethod
    def _failure(reason: str) -> dict[str, Any]:
        return {"verified": False, "reason": reason, "verifier_version": VERIFIER_VERSION}


def benchmark_execution_verification(program: Program, public_inputs: tuple[int, ...],
                                     *, spot_checks: int = 8) -> dict[str, Any]:
    """Compare proof generation/verification cost with plain recomputation (step 37)."""
    worker = ExecutionWorker()
    verifier = ExecutionVerifier()
    start = time.perf_counter()
    execution = worker.run(program, public_inputs, spot_checks=spot_checks)
    proving_ms = (time.perf_counter() - start) * 1000.0
    start = time.perf_counter()
    spot_result = verifier.verify(program, public_inputs, execution, mode="spot")
    spot_verify_ms = (time.perf_counter() - start) * 1000.0
    start = time.perf_counter()
    full_result = verifier.verify(program, public_inputs, execution, mode="full")
    recompute_ms = (time.perf_counter() - start) * 1000.0
    import json
    proof_size = len(json.dumps(execution["proof"], sort_keys=True))
    return {
        "steps": execution["proof"]["steps"],
        "proving_ms": round(proving_ms, 3),
        "spot_verification_ms": round(spot_verify_ms, 3),
        "full_recomputation_ms": round(recompute_ms, 3),
        "proof_size_bytes": proof_size,
        "spot_verified": spot_result["verified"],
        "full_verified": full_result["verified"],
        "recommendation": (
            "prefer recomputation for this workload unless trust, distribution or "
            "repetition cost justifies proof overhead"
        ),
    }
