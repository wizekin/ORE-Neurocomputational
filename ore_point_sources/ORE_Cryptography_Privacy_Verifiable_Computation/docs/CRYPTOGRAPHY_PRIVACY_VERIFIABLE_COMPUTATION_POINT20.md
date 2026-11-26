# ORE Cryptographic Trust and Privacy Layer (Point 20)

Cryptography, Privacy and Verifiable Computation — integrated algorithm implementation.

Source: "ORE Integrated Algorithm — Point 20: Cryptography, Privacy and Verifiable
Computation" (from Most Intelligent Systems 01-24, original sections 20.1–20.9).
All 64 steps of the integrated algorithm (phases A through J) are implemented in the
`ore_security` package; the operation registry in `ore_security/registry.py` maps
every step to its implementing module.

## What this layer is

An independent Cryptographic Trust and Privacy Layer (CTPL) that protects ORE's
cognitive architecture without entangling cryptographic policy with semantic
reasoning. The governing principle: ORE must know who acted, what they were allowed
to do, which data and program were involved, how integrity was checked, what privacy
guarantee applies, and which assumptions remain unproven.

The source technologies remain distinct because no single mechanism replaces another:

| Mechanism | Core idea | ORE role | Does not establish |
|---|---|---|---|
| Hash | h = H(x) | content integrity comparison | authenticity without trusted digest |
| Signature | Verify(pk, m, sigma) | artifact/message authenticity | truth or safety of signed content |
| Merkle tree | root over hashed leaves | batch inclusion, tamper evidence | semantic correctness |
| ZKP | prove R(x,w)=1 without revealing w | selected private constraints | correctness of specification |
| zkVM | proof of committed program execution | external computation verification | automatic privacy |
| FHE | compute on ciphertext | confidential selected computation | low cost or unrestricted operations |
| DP | bound influence of one privacy unit | private statistics | endpoint security |
| Secure aggregation | reveal aggregate, not inputs | collaborative telemetry | privacy beyond protocol assumptions |
| MPC | jointly compute f(x1..xn) | private multi-party computation | no leakage from permitted output |
| VDAF/Prio | verify private aggregate contributions | distributed measurement | general-purpose reasoning |
| PQC | quantum-resistant constructions | long-term migration readiness | endpoint security |

## Package layout (spec section 5)

```
ore_security/
  contracts.py        Section 5.1 typed interfaces (schema-validated, versioned)
  exceptions.py       Fail-closed exception hierarchy with audit error categories
  canonical.py        Canonical JSON serialization for every digest and signature
  crypto.py           Vetted-primitive facade (Ed25519, X25519, AES-GCM, HKDF)
  registry.py         All 64 source steps mapped to implementing modules
  pipeline.py         Ten-step security decision pipeline + ORE integration
  annotations.py      ore_logger SQLite registration and run persistence
  integration.py      ore_contracts catalog and module-manifest registration
  identity/           Node identities, scoped credentials, replay protection   (Phase B)
  policy/             Default-deny authorization, data classification          (Phase A)
  keys/               KMS adapter: lifecycle, envelope encryption, backups     (Phase C)
  transport/          Transport policy and mutually authenticated channels     (Phase C)
  signing/            Artifact manifests, signatures, layered verification     (Phase D)
  integrity/          Hash chains, Merkle trees, signed checkpoints            (Phase D)
  proofs/             ZKP statement registry and reference Schnorr adapter     (Phase E)
  zkvm/               Committed MiniVM programs with execution proofs          (Phase F)
  fhe/                Additive homomorphic encryption adapter (Paillier)      (20.3)
  privacy/            Gaussian/Laplace mechanisms, DP budget accounting       (Phase G)
  multiparty/         Secure aggregation, MPC, secret sharing, Beaver triples  (Phase H)
  vdaf/               Prio-style private validity checks and aggregation       (Phase H)
  pqc/                Algorithm agility, inventory, migration, hybrid envelopes(Phase I)
  audit/              Security events, tamper-evident ledger, incident response(Phase J)
```

## The security decision pipeline (spec section 2.2)

Every protected operation follows the same sequence in `SecurityDecisionPipeline`:

1. Identify the caller and authenticate its credential.
2. Resolve effective roles and scoped permissions.
3. Classify the requested resource and data sensitivity.
4. Check purpose, policy, time, context and operation constraints.
5. Establish an authenticated secure channel for remote communication.
6. Verify artifact/message signatures and integrity when applicable.
7. Execute the permitted operation in the appropriate isolation boundary.
8. Validate the result and any required proof.
9. Record an auditable event without leaking protected content.
10. Revoke, expire or rotate credentials when policy or incident response requires it.

The authorization predicate is default-deny:

```
A(i, o, r, c) = I(i) and P(i, o, r, c) and D(r) and T(c)
```

Missing policy data means deny. Authorization-service failure means deny
(fail-closed). Coordinator task priority is deliberately absent from the predicate:
`CoordinatorGuard` demonstrates that the ORE Main Brain Coordinator cannot override
security policy by raising priority. `SecuredRecordStore` gates USG updates on
artifact verification, authorization and semantic validation, because a signature
authenticates a signer, not the truth of a claim.

## Integration sequence status (spec section 6)

| Stage | Status | Where |
|---|---|---|
| 1. Identity, authorization, secure transport | implemented | `identity/`, `policy/`, `transport/` |
| 2. Signed artifacts and integrity evidence | implemented | `signing/`, `integrity/`, `audit/` |
| 3. KMS lifecycle and recovery | implemented (software adapter) | `keys/kms.py` |
| 4. Proof interface foundation | implemented (reference Schnorr) | `proofs/` |
| 5. External computation verification | implemented (MiniVM trace proofs) | `zkvm/` |
| 6. Privacy-preserving analytics | implemented | `privacy/`, `fhe/` |
| 7. Secure aggregation, MPC, VDAF | implemented (reference protocols) | `multiparty/`, `vdaf/` |
| 8. PQC migration readiness | implemented (agility + hybrid envelopes) | `pqc/` |

Integration with the repository: `annotations.py` registers every `ore_security`
symbol with the shared `ore_logger` SQLite catalog and persists pipeline runs;
`integration.py` registers the twelve section-5.1 interfaces with the shared
`ore_contracts` catalog and publishes the `ore.security.trust` module manifest.

## Honest scope and assumptions

This is a reference implementation exercising the full contract surface. The
following boundaries are deliberate and documented per spec section 9 (failure
modes) and the traceability note:

- **Primitives** come from the vetted `cryptography` library (Ed25519, X25519,
  AES-256-GCM, HKDF) and stdlib SHA-2. No hand-written signature or AEAD code.
- **The Schnorr proof adapter** is a reference discrete-log proof system
  (Fiat-Shamir, HVZK, safe-prime group). Production must bind a maintained ZKP
  library behind the same adapter contract.
- **The zkVM adapter** provides committed-program verifiable execution via trace
  Merkle roots and Fiat-Shamir spot checks. It is probabilistic in spot mode, not
  succinct, and explicitly does not provide privacy (`provides_privacy` is False).
- **The FHE adapter** is additive Paillier (partially homomorphic), a stand-in for
  vetted BFV/BGV/CKKS libraries; ciphertext-times-ciphertext is rejected.
- **The PQC module** provides algorithm agility, inventory, migration policy and
  hybrid envelopes. The Lamport one-time signature and mock KEM legs are envelope
  test vehicles; ML-KEM/ML-DSA/SLH-DSA primitives require a vetted provider.
- **The software KMS** protects keys in memory with audited use, canaries and
  encrypted backups; production requires an HSM- or OS-keystore-backed adapter.
- **Secure aggregation** tolerates server-only adversaries and n-1 participant
  collusion (Bonawitz-style bound with additive seed sharing); the VDAF validity
  check is affine AGREE over the full field, which binds honest binary
  measurements but not fully malicious clients (production Prio adds
  structured-ring per-coordinate checks).

## Acceptance tests (spec section 7)

Every acceptance criterion has a test. Run with `python -m pytest tests/test_ore_security_*.py`:

| Test file | Coverage |
|---|---|
| `test_ore_security_identity_policy.py` | unauthenticated/uns/expired/revoked/out-of-scope rejection, engine failure fail-closed, replay, peer identity, obsolete protocols |
| `test_ore_security_keys_artifacts.py` | at-rest encryption, key-material redaction, rotation history, revoked signers, artifact tamper/authorization/scope, hash chain, reordering, checkpoints, Merkle proofs |
| `test_ore_security_proofs_execution.py` | ZKP accept/reject, task binding, obsolete versions, version records, zkVM program/output binding, privacy disclaimer, semantic separation, cost benchmark |
| `test_ore_security_fhe_privacy.py` | FHE unsupported operations, authorized decryption, DP unit/adjacency, calibration, budget rejection, composition, utility and subgroups |
| `test_ore_security_multiparty_vdaf.py` | aggregation privacy, duplicates, round binding, dropout recovery, MPC reference match, threat model, VDAF validity, duplicates, cohort metadata, suppression |
| `test_ore_security_pqc.py` | unambiguous parsing, downgrade rejection, interoperability, hybrid legs, one-time enforcement, migration priority |
| `test_ore_security_audit_integration.py` | compromise drill, KMS outage recovery, SQLite secret scan, correlation ids, checkpoint restore, coordinator override, USG gating, proof failure, vetted libraries, ore_logger and ore_contracts integration |

## Quick start

```python
from ore_security import build_default_trust_stack

stack = build_default_trust_stack()
pipeline = stack["pipeline"]

result = pipeline.execute(
    action="usg.record.ingest",
    resource="usg://records/entry-1",
    purpose="record-ingest",
    credential_token=token,          # audience-bound, short-lived
    expected_audience="ore-service",
)
print(result.allowed, result.decision.reason_code)
```

See `examples/point20_ctpl_demo.py` for a full end-to-end demonstration covering
every phase, and `python -m pytest tests/test_ore_security_*.py` for the acceptance
suite. Scientific scope: this is an engineering trust layer, not a claim that any
cognitive or biological property is reproduced.
