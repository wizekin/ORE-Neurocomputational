"""End-to-end demonstration of the Point 20 Cryptographic Trust and Privacy Layer.

Runs one scenario through every phase: identity and policy, key management and
envelope encryption, artifact signing, the ten-step pipeline, ZKP and zkVM
verification, differential privacy, secure aggregation, MPC, VDAF, PQC readiness,
audit checkpoints and an incident drill. Run from the repository root:

    python examples/point20_ctpl_demo.py
"""
from __future__ import annotations

import numpy as np

from ore_security import (
    CoordinatorGuard,
    SecuredRecordStore,
    build_default_trust_stack,
    control_review_report,
    provider_report,
    registry_as_dicts,
)
from ore_security.contracts import PrivacyReleaseRequest, PrincipalKind
from ore_security.multiparty import (
    MPCProtocolEngine,
    AggregationParticipant,
    SecureAggregationServer,
    run_secure_aggregation_round,
)
from ore_security.pqc import CryptoInventory, MigrationManager, interoperability_test
from ore_security.policy import DataClassification, PolicyRule, ResourceDescriptor
from ore_security.privacy import PrivacyAccountant, dp_mean, evaluate_utility
from ore_security.proofs import SchnorrWitness
from ore_security.vdaf import PrioAggregationRound, PrioClient
from ore_security.zkvm import Instruction, Program, benchmark_execution_verification


def banner(title: str) -> None:
    print(f"\n=== {title} ===")


def main() -> None:
    print("ORE Cryptographic Trust and Privacy Layer -- Point 20 demo")
    print("provider:", provider_report()["library"], provider_report()["version"])
    print("spec steps implemented:", len(registry_as_dicts()))

    # -- Phase A/B: identities, policy, and the ten-step pipeline ----------------
    banner("Phase A/B: identity, policy, protected operation")
    stack = build_default_trust_stack()
    ids = stack["identity_service"]
    engine = stack["policy_engine"]
    kms = stack["kms"]
    signing = stack["signing"]
    pipeline = stack["pipeline"]

    ids.register_identity("svc-writer", PrincipalKind.SERVICE, roles=("WRITER",))
    ids.register_identity("svc-reader", PrincipalKind.SERVICE, roles=("READER",))
    writer_key = kms.generate_key("SIGNING", "Ed25519", key_id="key-usg-signer")
    engine.resource_registry.register(ResourceDescriptor(
        "usg://records", DataClassification.CONFIDENTIAL, "usg-owner",
        allowed_purposes=frozenset({"record-ingest"}),
    ))
    engine.add_rule(PolicyRule(
        "allow-writer-ingest", "svc-writer", frozenset({"usg.record.ingest"}),
        ("usg://records",), DataClassification.CONFIDENTIAL,
        required_purpose="record-ingest",
    ))
    writer_token, _ = ids.issue_credential("svc-writer", "ore-service",
                                           ("usg:write",), ttl=900)
    reader_token, _ = ids.issue_credential("svc-reader", "ore-service",
                                           ("usg:read",), ttl=900)

    allowed = pipeline.execute(
        action="usg.record.ingest", resource="usg://records/entry-1",
        purpose="record-ingest", credential_token=writer_token,
        expected_audience="ore-service", operation=lambda: "record-written",
    )
    print("authorized writer:", allowed.allowed, allowed.decision.reason_code,
          "| executed:", allowed.executed)
    denied = pipeline.execute(
        action="usg.record.ingest", resource="usg://records/entry-1",
        purpose="record-ingest", credential_token=reader_token,
        expected_audience="ore-service",
    )
    print("unauthorized reader denied:", not denied.allowed, denied.decision.reason_code)

    guard = CoordinatorGuard(pipeline)
    priority_attempt = guard.request_with_priority(
        task_priority=99, action="usg.record.ingest",
        resource="usg://records/entry-1", purpose="record-ingest",
        credential_token=reader_token, expected_audience="ore-service",
    )
    print("coordinator priority override rejected:", not priority_attempt.allowed)

    # -- Phase C: KMS, envelope encryption --------------------------------------
    banner("Phase C: key management and envelope encryption")
    secret_payload = b"confidential USG-derived aggregate"
    envelope = kms.envelope_encrypt(secret_payload, b"demo-aad")
    print("at-rest ciphertext != plaintext:",
          bytes.fromhex(envelope["ciphertext"]) != secret_payload)
    print("decrypt under policy:", kms.envelope_decrypt(envelope, b"demo-aad"))
    rewrapped = kms.rotate_kek_and_rewrap(envelope)
    print("KEK rotation rewrap still decrypts:",
          kms.envelope_decrypt(rewrapped, b"demo-aad") == secret_payload)

    # -- Phase D: signed artifacts, integrity, secured USG ingestion -------------
    banner("Phase D: signed artifacts and secured USG ingestion")
    payload = b'{"claim": "supported-by-evidence", "source": "audited-study"}'
    fields = signing.build_manifest("rec-demo", "1.0.0", payload,
                                    intended_use="usg-record",
                                    build_provenance="pipeline-demo")
    manifest = signing.sign(fields, writer_key.key_id)
    store = SecuredRecordStore(pipeline)
    stored = store.ingest(
        "rec-demo", payload, manifest, writer_token,
        purpose="record-ingest", resource_uri="usg://records/rec-demo",
        semantic_validator=lambda data: b"supported-by-evidence" in data,
        expected_use="usg-record", authorized_signers=frozenset({writer_key.key_id}),
    )
    print("semantically supported record stored:", stored["stored"])

    false_payload = b'{"claim": "the-moon-is-cheese"}'
    false_fields = signing.build_manifest("rec-false", "1.0.0", false_payload,
                                          intended_use="usg-record")
    false_manifest = signing.sign(false_fields, writer_key.key_id)
    refused = store.ingest(
        "rec-false", false_payload, false_manifest, writer_token,
        purpose="record-ingest", resource_uri="usg://records/rec-false",
        semantic_validator=lambda data: b"supported-by-evidence" in data,
        expected_use="usg-record", authorized_signers=frozenset({writer_key.key_id}),
    )
    print("intact but semantically false record refused:", not refused["stored"])

    # -- Phase E: zero-knowledge proof ------------------------------------------
    banner("Phase E: zero-knowledge proof of a credential secret")
    proof_service = stack["proof_service"]
    witness = SchnorrWitness(20261123)
    y = witness.public_value(proof_service.adapter.group)
    statement = proof_service.build_statement(
        "stmt-demo", {"y": y}, task_id="task-demo", task_nonce="n-demo", ttl=120.0,
    )
    proof_record = proof_service.prove(statement, witness)
    verification = proof_service.verify(proof_record, statement)
    print("ZKP verified:", verification.verified, "|",
          verification.reason, "| verifier:", verification.verifier_version)

    # -- Phase F: verifiable execution ------------------------------------------
    banner("Phase F: committed-program execution proof")
    program = Program(
        instructions=(
            Instruction("PUSHP", 0), Instruction("PUSHC", 0), Instruction("MUL"),
            Instruction("PUSHC", 1), Instruction("ADD"), Instruction("POPM", 0),
            Instruction("HALT"),
        ),
        constants=(11, 5),
    )
    benchmark = benchmark_execution_verification(program, (9,), spot_checks=6)
    print("execution steps:", benchmark["steps"],
          "| outputs verified:", benchmark["full_verified"],
          "| spot verified:", benchmark["spot_verified"])
    print("costs: proving", benchmark["proving_ms"], "ms; verification",
          benchmark["spot_verification_ms"], "ms; recomputation",
          benchmark["full_recomputation_ms"], "ms")
    print("zkVM privacy claim: none (verifiable execution only)")

    # -- Phase G: differential privacy ------------------------------------------
    banner("Phase G: differentially private telemetry")
    accountant = PrivacyAccountant(epsilon_budget=1.5, delta_budget=1e-5)
    rng = np.random.default_rng(7)
    module_latencies = np.clip(rng.normal(240.0, 30.0, size=800), 0.0, 600.0)
    release = dp_mean(list(module_latencies), clip_lower=0.0, clip_upper=600.0,
                      epsilon=0.7, delta=1e-6, rng=rng)
    request = PrivacyReleaseRequest(
        "rel-demo", "ore-instance", "add/remove one ORE instance", "mean module latency",
        release["mechanism"]["sensitivity"], "Gaussian", 0.7, 1e-6,
        "module telemetry", "telemetry-v1", calibration_reviewed=True,
    )
    record = accountant.request_release(request)
    utility = evaluate_utility(float(np.mean(module_latencies)),
                               [release["released_value"]])
    print("DP release approved:", record.approved,
          "| composed epsilon:", round(record.composed_epsilon, 3))
    print("released mean:", round(release["released_value"], 2),
          "| true mean:", round(float(np.mean(module_latencies)), 2),
          "| error:", round(abs(utility["bias"]), 3))

    # -- Phase H: secure aggregation, MPC, VDAF ---------------------------------
    banner("Phase H: multiparty protocols")
    participants = [AggregationParticipant(f"ore-node-{i}") for i in (1, 2, 3, 4)]
    telemetry = {p.participant_id: [120, 4] for p in participants}
    server = SecureAggregationServer(min_cohort_size=3)
    aggregate = run_secure_aggregation_round(participants, telemetry, server, "telemetry-r1")
    print("secure aggregation output:", aggregate.output,
          "| cohort:", aggregate.cohort_size,
          "| individuals hidden:", all(
              server.server_visible_data("telemetry-r1")[p.participant_id] != [120, 4]
              for p in participants))

    mpc = MPCProtocolEngine(rng=rng)
    session = mpc.create_session("mpc-demo", ["alice", "bob", "carol"],
                                 permitted_outputs=("sum", "inner_product"))
    total = mpc.secure_sum(session, {"alice": [10, 20], "bob": [30, 40], "carol": [50, 60]})
    print("MPC secure sum:", total["output"], "(reference [90, 120])")
    inner = mpc.secure_inner_product(session, "alice", [2, 3], "bob", [4, 5])
    print("MPC inner product:", inner["output"], "(reference 23)")

    prio_client = PrioClient(rng=rng)
    vdaf_round = PrioAggregationRound("vdaf-demo", min_cohort_size=3, rng=rng)
    for index in range(5):
        vdaf_round.submit(f"client-{index}", prio_client.encode([1, 0, 1, 1]))
    vdaf_aggregate = vdaf_round.aggregate()
    print("VDAF aggregate:", vdaf_aggregate.output,
          "| protocol:", vdaf_aggregate.protocol_ref,
          "| cohort:", vdaf_aggregate.cohort_size)

    # -- Phase I: post-quantum readiness ----------------------------------------
    banner("Phase I: post-quantum migration readiness")
    inventory = CryptoInventory()
    inventory.record_usage("kms.transport", "X25519", purpose="key-establishment",
                           data_lifetime_years=25.0)
    inventory.record_usage("signing.release", "Ed25519", purpose="signature")
    manager = MigrationManager(inventory)
    print("migration priorities:", [(p["component"], p["priority"])
                                    for p in manager.prioritize()])
    print("hybrid interop test passed:", interoperability_test()["passed"])

    # -- Phase J: audit, checkpoints, incident drill -----------------------------
    banner("Phase J: audit integrity and incident response")
    checkpoint = stack["audit"].checkpoint()
    verification_report = stack["audit"].verify_history()
    print("audit events:", stack["ledger"].event_count(),
          "| checkpoint range:", checkpoint["range"],
          "| history valid:", verification_report["valid"])
    from ore_security.audit.service import IncidentResponse
    kms.generate_key("SIGNING", "Ed25519", key_id="key-svc-reader-signing")
    incident = IncidentResponse(ids, kms, stack["audit"])
    drill = incident.compromise_principal("svc-reader")
    print("compromise drill contained:", drill["actions"])
    print("\ncontrol review limitations (step 64):")
    for limitation in control_review_report()["known_limitations"][:3]:
        print(" -", limitation)
    print("\nPoint 20 CTPL demo completed successfully.")


if __name__ == "__main__":
    main()
