# ORE Point 20 — Cryptography, Privacy and Verifiable Computation

Algorithm implementation package for the ORE-Neurocomputational repository:
the ORE Cryptographic Trust and Privacy Layer (CTPL), implementing all 64 steps
(phases A–J) of "ORE Integrated Algorithm — Point 20: Cryptography, Privacy and
Verifiable Computation".

## Contents

| Path | Description |
|---|---|
| `ore_security/` | The algorithm package: 52 Python modules across 13 subpackages |
| `tests/test_ore_security_*.py` | 94 acceptance tests covering the spec section-7 checklist |
| `docs/CRYPTOGRAPHY_PRIVACY_VERIFIABLE_COMPUTATION_POINT20.md` | Architecture, assumptions and acceptance-test map |
| `examples/point20_ctpl_demo.py` | End-to-end demonstration of every phase |

## Installation into the ORE-Neurocomputational repository

1. Copy the folders so they sit at the repository root:

   ```
   ORE-Neurocomputational/
     ore_security/            <- copy this folder
     tests/test_ore_security_*.py   <- copy these files into tests/
     docs/CRYPTOGRAPHY_PRIVACY_VERIFIABLE_COMPUTATION_POINT20.md
     examples/point20_ctpl_demo.py
   ```

2. Add `"ore_security*"` to the `include` list in `pyproject.toml` under
   `[tool.setuptools.packages.find]` (one line; it is the only existing file that
   needs modification).

3. Requirements: Python 3.10+, `numpy`, and the vetted `cryptography` library
   (`pip install cryptography`) for Ed25519 / X25519 / AES-256-GCM / HKDF
   primitives. `pytest` for the test suite.

4. Verify:

   ```
   python -m pytest tests/test_ore_security_identity_policy.py \
     tests/test_ore_security_keys_artifacts.py \
     tests/test_ore_security_proofs_execution.py \
     tests/test_ore_security_fhe_privacy.py \
     tests/test_ore_security_multiparty_vdaf.py \
     tests/test_ore_security_pqc.py \
     tests/test_ore_security_audit_integration.py

   PYTHONPATH=. python examples/point20_ctpl_demo.py
   ```

## ORE integration points (no existing module is modified)

- `ore_security.annotations.register_annotations()` indexes every symbol in the
  shared `ore_logger` SQLite catalog; `record_pipeline_run_to_sqlite()` persists
  protected-operation runs as linked events.
- `ore_security.integration.register_with_ore()` registers the twelve section-5.1
  typed interfaces with the shared `ore_contracts` catalog and publishes the
  `ore.security.trust` module manifest.
- `ore_security.pipeline` provides the ten-step security decision pipeline,
  the `CoordinatorGuard` (priority cannot override policy) and the
  `SecuredRecordStore` (USG updates require security and semantic validation).

## Scope statement

Reference implementation: primitives come from the vetted `cryptography`
library; protocol-level constructions (Schnorr ZKP, MiniVM execution proofs,
Paillier, masking, secret sharing, Prio, Lamport reference OTS) are documented
reference adapters behind typed contracts and must be backed by maintained
libraries in production. See the "Honest scope and assumptions" section of the
documentation file for the full list.
