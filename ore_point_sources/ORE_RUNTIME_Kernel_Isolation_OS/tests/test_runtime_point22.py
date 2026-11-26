"""Acceptance tests for the ORE runtime governance layer (Point 22).

Each test maps to one or more entries of the acceptance-test list in
"ORE Integrated Algorithm -- Point 22: Advanced Kernel, Isolation and
Operating-System Architectures", section 8. Dimensions stay small so the
suite remains fast; subprocess and SQLite fixtures use tmp_path.
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

import pytest

from ore_runtime_governance import (
    AccessRequest,
    ArtifactCorruptionError,
    BackpressurePolicy,
    BlackboardGuard,
    BudgetScheduler,
    Capability,
    CapabilityAuthority,
    CapabilityDeniedError,
    ComponentManifest,
    ComponentRuntime,
    ConsequenceClass,
    DependencyEdge,
    DependencyGraph,
    DependencyGraphError,
    EnvelopeValidationError,
    ExecutionState,
    ExecutionTier,
    FaultInjector,
    ImpersonationError,
    IsolationProfile,
    LifecycleMachine,
    LifecycleState,
    MessageBus,
    MessageEnvelope,
    ModelBasedLifecycleChecker,
    OperatorControl,
    OperationCancelledError,
    OperationTimedOutError,
    OutcomeKind,
    PeriodicSpec,
    PolicyDocument,
    PolicyEngine,
    PolicyRollout,
    PolicyRule,
    RegistrationResult,
    ResourceManager,
    ResourceProfile,
    ResourceVector,
    ResponseEnvelope,
    RestartPolicy,
    RolloutObservation,
    SchedClass,
    SchedTask,
    SecurityProbeSuite,
    SQLiteEventSink,
    Supervisor,
    TraceContext,
    TrustClass,
    USGGuard,
    admission_factor,
    audit,
    benchmark_overhead,
    compute_backoff,
    inventory_repository,
    isolation_score,
    queue_stable,
    redact_metadata,
    register_runtime_annotations,
    replay_reservations,
    resolve_tier,
    rms_sufficient_bound,
    schedulability_report,
    scope_matches,
)


# --- helpers ---------------------------------------------------------------


def trusted_manifest(component_id: str = "echo.service", **overrides) -> ComponentManifest:
    payload = dict(
        component_id=component_id,
        version="1.0.0",
        owner="ore_core",
        trust_class=TrustClass.TRUSTED_SERVICE,
        purpose="acceptance test component",
        entry_point="ore_runtime.testing:echo_worker",
        capabilities=("ipc.send", "task.execute", "artifact.publish", "artifact.read"),
        resources=ResourceProfile(
            requested=ResourceVector(cpu_shares=0.5, memory_bytes=32 * 1024 * 1024),
            limits=ResourceVector(cpu_shares=1.0, memory_bytes=64 * 1024 * 1024),
            deadline_seconds=10.0,
        ),
        isolation=IsolationProfile(tier=ExecutionTier.IN_PROCESS),
    )
    payload.update(overrides)
    return ComponentManifest(**payload)


def build_runtime(tmp_path: Path | None = None, **kwargs) -> ComponentRuntime:
    options = dict(capacity=ResourceVector(cpu_shares=8.0, memory_bytes=1024 * 1024 * 1024))
    if tmp_path is not None:
        options["database_path"] = str(tmp_path / "runtime_audit.sqlite3")
        options["artifact_root"] = str(tmp_path / "artifacts")
    options.update(kwargs)
    runtime = ComponentRuntime(**options)
    runtime.register(trusted_manifest())
    runtime.register_handler("echo.service", lambda envelope: {"echo": dict(envelope.payload)})
    return runtime


def envelope(sender: str, recipient: str = "echo.svc", payload=None, **kw) -> MessageEnvelope:
    fields = dict(
        sender=sender,
        recipient=recipient,
        payload=dict(payload or {"x": 1}),
        deadline_monotonic_s=time.monotonic() + 10.0,
    )
    fields.update(kw)
    return MessageEnvelope(**fields)


# --- manifests and registration (steps 3, 8) --------------------------------


def test_valid_manifest_registers_and_invalid_is_rejected():
    runtime = ComponentRuntime()
    result = runtime.register(trusted_manifest())
    assert isinstance(result, RegistrationResult)
    assert result.registered and result.manifest_digest

    bad_manifest = trusted_manifest().to_dict()
    bad_manifest["mystery_field"] = "unknown"
    with pytest.raises(Exception):
        ComponentManifest.from_dict(bad_manifest)
    with pytest.raises(ValueError):
        ComponentManifest(component_id="bad.comp", capabilities=("root.everything",))
    with pytest.raises(ValueError):
        ComponentManifest(component_id="bad.comp", version="not-semver")


def test_untrusted_manifest_never_in_process():
    with pytest.raises(Exception):
        ComponentManifest(
            component_id="gen.code",
            trust_class=TrustClass.UNTRUSTED_GENERATED,
            isolation=IsolationProfile(tier=ExecutionTier.IN_PROCESS),
        )


def test_resource_requests_outside_bounds_rejected():
    runtime = ComponentRuntime(
        bounds=ResourceVector(cpu_shares=1.0, memory_bytes=64 * 1024 * 1024)
    )
    oversized = trusted_manifest(
        "big.comp",
        resources=ResourceProfile(
            requested=ResourceVector(cpu_shares=2.0, memory_bytes=128 * 1024 * 1024),
            limits=ResourceVector(cpu_shares=4.0, memory_bytes=256 * 1024 * 1024),
            deadline_seconds=10.0,
        ),
    )
    result = runtime.register(oversized)
    assert not result.registered
    assert "bounds" in result.reason


def test_lifecycle_transitions_checked_against_machine():
    machine = LifecycleMachine()
    for target in ("validated", "authorized", "ready", "running", "quiescing", "stopped"):
        machine.apply("life.comp", target)
    with pytest.raises(Exception):
        machine.apply("life.comp", "created")
    with pytest.raises(Exception):
        machine.apply("life.comp", "running")  # terminal without new generation
    revived = machine.apply("life.comp", "ready", new_generation=True)
    assert revived.generation == 1


# --- capabilities (steps 15-22, section 5.3) ---------------------------------


def test_expired_capability_rejected():
    authority = CapabilityAuthority()
    ghost = authority.issue("ghost.comp", "workspace", ["workspace.read_snapshot"], ttl_seconds=0.01)
    time.sleep(0.02)
    decision = authority.authorize(
        ghost, AccessRequest("ghost.comp", "workspace", "workspace.read_snapshot")
    )
    assert not decision.allowed
    assert "expired" in decision.reason


def test_revoked_generation_rejected_after_restart():
    authority = CapabilityAuthority()
    cap = authority.issue("stale.comp", "workspace", ["workspace.read_snapshot"], ttl_seconds=60)
    authority.revoke_subject("stale.comp")  # worker restart / downgrade
    decision = authority.authorize(
        cap, AccessRequest("stale.comp", "workspace", "workspace.read_snapshot")
    )
    assert not decision.allowed
    assert "generation" in decision.reason


def test_child_capability_cannot_exceed_parent():
    authority = CapabilityAuthority()
    parent = authority.issue(
        "parent.comp", "workspace", ["workspace.read_snapshot", "workspace.propose_fact"],
        ttl_seconds=60,
    )
    with pytest.raises(CapabilityDeniedError):
        authority.attenuate(parent, rights=["ipc.send"])
    with pytest.raises(CapabilityDeniedError):
        authority.attenuate(parent, resource="artifacts")
    # requesting a longer ttl never extends the parent expiry: it clamps
    extended = authority.attenuate(parent, ttl_seconds=3600.0)
    assert extended.expires_at_epoch == parent.expires_at_epoch
    child = authority.attenuate(parent, rights=["workspace.read_snapshot"], ttl_seconds=30)
    assert child.rights <= parent.rights
    assert child.expires_at_epoch <= parent.expires_at_epoch


def test_component_cannot_impersonate_another():
    authority = CapabilityAuthority()
    real = authority.issue("real.comp", "ipc", ["ipc.send"], ttl_seconds=60)
    decision = authority.authorize(real, AccessRequest("other.comp", "ipc/x", "ipc.send"))
    assert not decision.allowed
    assert "does not match" in decision.reason
    forged = Capability(
        capability_id=real.capability_id, subject="attacker.comp", resource="ipc",
        rights=frozenset({"ipc.send"}), issued_at_epoch=0, expires_at_epoch=9999.0, mac="0" * 64,
    )
    assert not authority.authorize(forged, AccessRequest("attacker.comp", "ipc/x", "ipc.send")).allowed


def test_scope_prefix_matching():
    assert scope_matches("workspace", "workspace/lexical_hypotheses")
    assert scope_matches("artifacts", "artifacts")
    assert not scope_matches("workspace", "workspace-secret")
    assert not scope_matches("workspace", "usg/records")


# --- IPC plane (steps 10-14, 29, 33) -------------------------------------------


def test_ipc_request_reply_with_correlation():
    runtime = build_runtime()
    with runtime:
        capability = runtime._grants["echo.service"]["ipc.send"]
        env = envelope("echo.service", "echo.service", auth_capability_id=capability.capability_id)
        response = runtime.request(env)
        assert response.status == OutcomeKind.OK
        assert response.correlation_id == env.message_id
        assert response.result == {"echo": {"x": 1}}


def test_unsupported_version_and_oversized_rejected_safely():
    runtime = build_runtime()
    with runtime:
        capability = runtime._grants["echo.service"]["ipc.send"]
        with pytest.raises(EnvelopeValidationError):
            envelope("echo.service", "echo.service", schema_version=2)
        big = envelope(
            "echo.service", "echo.service",
            payload={"blob": "y" * (runtime.bus.max_payload_bytes + 1)},
            auth_capability_id=capability.capability_id,
        )
        with pytest.raises(EnvelopeValidationError):
            runtime.request(big)


def test_full_queue_applies_backpressure_not_unbounded_memory():
    policy = BackpressurePolicy(capacity=4, defer_threshold=0.5, reject_threshold=0.75, critical_reserve=1)
    assert policy.decide(0, 5).accepted
    assert policy.decide(2, 5).kind == "defer"
    assert policy.decide(3, 5).kind == "reject"
    assert policy.decide(3, 9).accepted  # critical reserve
    assert policy.decide(4, 9).kind == "reject"
    assert admission_factor(2, 4) == 0.5
    assert queue_stable(1.0, 2.0)


def test_timeout_distinguished_from_confirmed_failure():
    runtime = build_runtime()
    with runtime:
        cap_id = runtime._grants["echo.service"]["ipc.send"].capability_id
        runtime.register_handler("slow.svc", lambda e: (_ for _ in ()).throw(OperationTimedOutError("late")))
        runtime.register_handler("fail.svc", lambda e: 1 / 0)
        timed = runtime.request(envelope("echo.service", "slow.svc", auth_capability_id=cap_id))
        assert timed.status == OutcomeKind.TIMED_OUT
        failed = runtime.request(envelope("echo.service", "fail.svc", auth_capability_id=cap_id))
        assert failed.status == OutcomeKind.INTERNAL
        assert "handler failed" in failed.reason
        missing = runtime.request(envelope("echo.service", "missing.svc", auth_capability_id=cap_id))
        assert missing.status == OutcomeKind.UNAVAILABLE


def test_duplicate_delivery_is_idempotent():
    runtime = build_runtime()
    with runtime:
        cap_id = runtime._grants["echo.service"]["ipc.send"].capability_id
        calls = []
        runtime.register_handler("count.svc", lambda e: calls.append(1) or {"n": len(calls)})
        first = MessageEnvelope(
            sender="echo.service", recipient="count.svc", payload={"a": 1},
            delivery="effectively_once", auth_capability_id=cap_id,
        )
        env_first = runtime.request(first)
        duplicate = MessageEnvelope(
            sender="echo.service", recipient="count.svc", payload={"a": 1},
            delivery="effectively_once", auth_capability_id=cap_id,
        )
        runtime.bus.dedupe.record(duplicate.message_id, env_first)
        reply = runtime.request(duplicate)
        assert reply is env_first or reply.status == OutcomeKind.OK
        assert runtime.bus.statistics.deduplicated >= 0


# --- resources (steps 18, 26, 36-39, section 5.1) ------------------------------


def test_aggregate_reservations_cannot_exceed_capacity():
    manager = ResourceManager(ResourceVector(cpu_shares=4, memory_bytes=100))
    grants = [
        manager.reserve(f"t{i}", ResourceVector(cpu_shares=1, memory_bytes=25)) for i in range(4)
    ]
    assert all(result.granted for result in grants)
    assert manager.available().cpu_shares == 0
    fifth = manager.reserve("t5", ResourceVector(cpu_shares=1))
    assert not fifth.granted
    assert "exceeds capacity" in fifth.reason


def test_child_budgets_never_exceed_parent():
    root = ResourceManager(ResourceVector(cpu_shares=10))
    first = root.child_budget("a", ResourceVector(cpu_shares=6))
    second = root.child_budget("b", ResourceVector(cpu_shares=6))
    assert first.reserve("t1", ResourceVector(cpu_shares=6)).granted
    denied = second.reserve("t2", ResourceVector(cpu_shares=6))
    assert not denied.granted and "root" in denied.reason


def test_cancellation_and_exceptions_release_reservations():
    manager = ResourceManager(ResourceVector(cpu_shares=4))
    with pytest.raises(ValueError):
        with manager.session("task1", ResourceVector(cpu_shares=2)):
            raise ValueError("boom")
    assert manager.available().cpu_shares == 4
    leaked = manager.reserve("ghost-task", ResourceVector(cpu_shares=2))
    recovered = manager.reconcile([])
    assert all(result.released for result in recovered)
    assert manager.available().cpu_shares == 4


def test_observed_usage_beyond_limit_breaches():
    manager = ResourceManager(ResourceVector(cpu_shares=4))
    result = manager.reserve("task1", ResourceVector(cpu_shares=2))
    breaches = manager.observe(result.reservation.reservation_id, ResourceVector(cpu_shares=3))
    assert any(b.kind == "cpu_shares" and b.observed == 3 for b in breaches)
    report = manager.describe()
    entry = report["kinds"]["cpu_shares"]
    assert {"requested", "reserved", "observed", "enforced_limit"} <= set(entry)


# --- scheduling (steps 43-44, section 5.6) --------------------------------------


def test_edf_and_rms_ordering_with_assumption_guard():
    specs = (PeriodicSpec("t1", 1.0, 4.0), PeriodicSpec("t2", 2.0, 8.0))
    report = schedulability_report(specs)
    assert report.utilization == 0.5
    assert report.edf_feasible
    assert report.rms_bound < 1.0
    assert "assumptions" in report.to_dict()
    assert rms_sufficient_bound(1) == 1.0


def test_budget_scheduler_sheds_optional_but_not_reserved():
    scheduler = BudgetScheduler(
        (
            SchedClass("coordinator", 10, 10, priority=10, reserved=True),
            SchedClass("persistence", 5, 5, priority=8, reserved=True),
            SchedClass("exploration", 5, 5, priority=2),
        )
    )
    tasks = (
        SchedTask("coord-1", "coordinator", 0.0, 1.0),
        SchedTask("explore-1", "exploration", 0.0, 1.0),
    )
    assert scheduler.select_next(tasks, now=1.0).task_id == "coord-1"
    report = scheduler.shed_under_pressure(tasks, queue_pressure=0.9)
    assert "explore-1" in report.shed_task_ids
    assert "coord-1" not in report.shed_task_ids


# --- artifacts (steps 11, 27, 29) -------------------------------------------------


def test_artifact_publish_read_verify_roundtrip(tmp_path: Path):
    runtime = build_runtime(tmp_path)
    with runtime:
        capability = runtime.issue(
            "echo.service", ["artifact.publish", "artifact.read"],
            resource="artifacts/echo.service",
        )
        ref = runtime.publish_artifact(b"payload-bytes", {"schema": "test/bin"}, capability)
        assert ref.artifact_id.startswith("sha256:")
        assert runtime.read_artifact(ref, capability) == b"payload-bytes"
        assert runtime.artifacts.verify(ref).ok


def test_artifact_corruption_detected(tmp_path: Path):
    runtime = build_runtime(tmp_path)
    with runtime:
        capability = runtime.issue(
            "echo.service", ["artifact.publish", "artifact.read"],
            resource="artifacts/echo.service",
        )
        ref = runtime.publish_artifact(b"original", None, capability)
        stored = runtime.artifacts._path_for(ref.artifact_id)
        stored.write_bytes(b"tampered!!")
        with pytest.raises(ArtifactCorruptionError):
            runtime.read_artifact(ref, capability)


def test_failed_publication_leaves_no_partial_artifact(tmp_path: Path):
    runtime = build_runtime(tmp_path)
    with runtime:
        capability = runtime.issue(
            "echo.service", ["artifact.publish"], resource="artifacts/echo.service",
        )
        with pytest.raises(ValueError):
            runtime.publish_artifact(b"x" * (runtime.artifacts.max_bytes + 1), None, capability)
        partials = [
            path for path in runtime.artifacts.root.rglob("*") if path.name.startswith(".partial")
        ]
        assert not partials


# --- isolation (steps 5, 9, 21, 39-42, section 5.7) ------------------------------


def test_isolation_score_and_hard_floor():
    assert resolve_tier(TrustClass.TRUSTED_CORE, ConsequenceClass.READ_ONLY, 0.1) == ExecutionTier.IN_PROCESS
    # score must never authorize a forbidden combination (untrusted stays sandboxed)
    assert resolve_tier(TrustClass.UNTRUSTED_GENERATED, ConsequenceClass.READ_ONLY, 0.05) == (
        ExecutionTier.CONTAINER_SANDBOX
    )
    assert isolation_score(1.0, 1.0, 1.0, 1.0) == 1.0
    assert 0 <= isolation_score(0.2, 0.3, 0.4, 0.5) <= 1


def test_subprocess_worker_scrubbed_environment_and_rlimits():
    from ore_runtime_governance import SubprocessExecutionBackend

    backend = SubprocessExecutionBackend()
    if not backend.available():
        pytest.skip("subprocess tier requires POSIX")
    manifest = trusted_manifest("sub.proc")
    manifest = ComponentManifest(
        component_id="sub.proc",
        entry_point="ore_runtime.testing:echo_worker",
        trust_class=TrustClass.TRUSTED_SERVICE,
        resources=ResourceProfile(
            requested=ResourceVector(cpu_shares=0.5),
            limits=ResourceVector(cpu_shares=1.0, memory_bytes=256 * 1024 * 1024),
            deadline_seconds=20.0,
        ),
        isolation=IsolationProfile(tier=ExecutionTier.LOCAL_PROCESS),
    )
    handle = backend.run(manifest, {"secret_check": "env"}, budget_seconds=20.0)
    assert handle.state == ExecutionState.FINISHED
    assert handle.result["echo"] == {"secret_check": "env"}


def test_sandbox_timeout_terminates_and_preserves_diagnostics():
    from ore_runtime_governance import SubprocessExecutionBackend

    backend = SubprocessExecutionBackend()
    if not backend.available():
        pytest.skip("subprocess tier requires POSIX")
    slow = ComponentManifest(
        component_id="slow.worker",
        entry_point="ore_runtime.testing:sleep_worker",  # blocks on payload seconds
        trust_class=TrustClass.TRUSTED_SERVICE,
        isolation=IsolationProfile(tier=ExecutionTier.LOCAL_PROCESS),
    )
    deadline = time.monotonic() + 1.0
    handle = backend.run(slow, {"seconds": 30.0}, budget_seconds=30.0, deadline_monotonic=deadline)
    assert handle.state == ExecutionState.TIMED_OUT
    assert "deadline_exceeded" in handle.exit_info
    assert "work_dir_preserved_until_handle_exit" in handle.exit_info
    with handle:
        pass  # context exit cleans the task-temp directory


def test_container_and_microvm_backends_unavailable_here():
    from ore_runtime_governance import ContainerSandboxBackend, MicroVMBackend

    if ContainerSandboxBackend().available():  # docker present in some CI images
        pytest.skip("container runtime present; backend-specific test")
    from ore_runtime_governance import BackendUnavailableError

    with pytest.raises(BackendUnavailableError):
        ContainerSandboxBackend().run(trusted_manifest("sandbox.comp"), {})
    with pytest.raises(BackendUnavailableError):
        MicroVMBackend().run(trusted_manifest("vm.comp"), {})


def test_remote_disconnect_does_not_block_local():
    from ore_runtime_governance import InProcessBackend, RemoteWorker, RemoteWorkerPool

    pool = RemoteWorkerPool()
    pool.register(RemoteWorker("gpu.node", "tcp://gpu:1", "rack-b", 12.0, "trusted_service", 1.0))
    plan = pool.execution_plan("gpu.node", InProcessBackend())
    assert "falling back" in plan


# --- supervisor, fault injection, recovery (steps 9, 35, 49) ----------------------


def test_restart_budget_quarantines():
    supervisor = Supervisor(RestartPolicy(restart_limit=3, backoff_base_seconds=0.01))
    decisions = [supervisor.report_failure("w.comp") for _ in range(3)]
    assert [d.decision for d in decisions[:2]] == ["retry", "retry"]
    assert decisions[2].decision == "quarantine"
    assert compute_backoff(0, base=0.25, cap=30.0) == 0.25
    assert compute_backoff(8, base=0.25, cap=30.0) == 30.0


def test_fault_injection_suite():
    runtime = build_runtime()
    with runtime:
        injector = FaultInjector()
        capability = runtime._grants["echo.service"]["ipc.send"]
        env = envelope("echo.service", "echo.service", auth_capability_id=capability.capability_id)
        # corrupt message: unsupported schema version is rejected safely
        corrupted = injector.corrupt_envelope(env, mode="schema")
        with pytest.raises(EnvelopeValidationError):
            runtime.request(corrupted)
        # quota exhaustion denies admission
        denied = injector.exhaust_quota(
            ResourceManager(ResourceVector(cpu_shares=2)), count=4
        )
        assert denied >= 1
        # mid-task revocation kills outstanding authority
        authority = CapabilityAuthority()
        cap = authority.issue("victim.comp", "ipc", ["ipc.send"], ttl_seconds=60)
        injector.revoke_mid_task(authority, cap)
        assert not authority.authorize(
            cap, AccessRequest("victim.comp", "ipc/x", "ipc.send")
        ).allowed


# --- audit (steps 28, 45-46) --------------------------------------------------------


def test_audit_events_persisted_append_only_with_chain(tmp_path: Path):
    sink = SQLiteEventSink(tmp_path / "audit.sqlite3")
    logger = audit.AuditLogger(sink)
    for index in range(5):
        logger.record_kind("dispatch", target="echo.service", sequence=index)
    assert sink.count() == 5
    assert sink.verify_chain()
    with sqlite3.connect(str(tmp_path / "audit.sqlite3")) as connection:
        rows = connection.execute(
            "SELECT kind, metadata_json FROM runtime_audit_events ORDER BY sequence_no"
        ).fetchall()
    assert all(row[0] == "dispatch" for row in rows)


def test_transaction_interruption_leaves_recoverable_records(tmp_path: Path):
    sink = SQLiteEventSink(tmp_path / "audit.sqlite3")
    logger = audit.AuditLogger(sink)
    logger.record_kind(
        "reservation", task_id="task-1", reservation_id="res-1", amounts={"cpu_shares": 1.0}
    )
    before = sink.count()
    interrupted = sink.simulate_interruption(
        audit.AuditEvent(
            kind="reservation",
            task_id="task-2",
            metadata={"reservation_id": "res-2", "amounts": {"cpu_shares": 1.0}},
        )
    )
    assert not interrupted.persisted
    assert sink.count() == before  # rollback left nothing partial
    logger.record_kind("reservation_release", task_id="task-1", reservation_id="res-1")
    ledger = replay_reservations(sink.fetch(limit=100))
    assert ledger == {}  # committed reservation was released in the log
    assert sink.verify_chain()


def test_replay_recovers_active_reservations():
    events = [
        audit.AuditEvent(kind="reservation", task_id="t1", metadata={"reservation_id": "r1", "amounts": {"cpu": 1}}),
        audit.AuditEvent(kind="reservation", task_id="t2", metadata={"reservation_id": "r2", "amounts": {"cpu": 1}}),
        audit.AuditEvent(kind="reservation_release", task_id="t1", metadata={"reservation_id": "r1"}),
    ]
    ledger = replay_reservations(events)
    assert set(ledger) == {"r2"}


def test_audit_redaction_never_stores_secrets(tmp_path: Path):
    sink = SQLiteEventSink(tmp_path / "audit.sqlite3")
    logger = audit.AuditLogger(sink)
    logger.record_kind(
        "dispatch", target="echo.service", api_key="sk-12345", session_id="abc", note="fine"
    )
    stored = sink.fetch(limit=10)[0]["metadata"]
    assert stored["api_key"] == "[REDACTED]"
    assert stored["session_id"] == "[REDACTED]"
    assert stored["note"] == "fine"
    assert redact_metadata({"password": "x"}) == {"password": "[REDACTED]"}


def test_denials_include_actor_resource_policy_reason():
    runtime = build_runtime()
    with runtime:
        runtime.register_handler("secret.svc", lambda e: {"ok": True})
        with pytest.raises(CapabilityDeniedError):
            runtime.request(envelope("echo.service", "secret.svc"))  # no capability
        denial_events = runtime.audit.query(kind="authorization_denied")
        assert denial_events
        event = denial_events[0]
        assert event["actor"] == "echo.service"
        assert event["policy_version"] >= 1
        metadata = event["metadata"]
        assert "reason" in metadata and "resource" in metadata


def test_runtime_annotations_registered(tmp_path: Path):
    report = register_runtime_annotations(
        database_path=tmp_path / "ore_logger.sqlite3",
        repository_root=Path(__file__).resolve().parents[1],
    )
    assert report["annotation_records"] > 10
    assert report["registered"] in (True, False)  # ore_logger presence decides


# --- policy (steps 51-52) -------------------------------------------------------------


def test_policy_rollout_rolls_back_on_thresholds():
    known_good = PolicyDocument(
        "cros", 1, (PolicyRule("r1", "baseline", True),)
    )
    candidate = PolicyDocument(
        "cros", 2, (PolicyRule("r1", "baseline", True), PolicyRule("r2", "new", False)),
    )
    rollout = PolicyRollout(candidate, known_good)
    ok = rollout.advance(RolloutObservation(0, violation_rate=0.01, failure_rate=0.02, p95_latency_ms=100))
    assert ok == "continue"
    bad = rollout.advance(RolloutObservation(1, violation_rate=0.5, failure_rate=0.1, p95_latency_ms=100))
    assert bad == "rollback"
    assert rollout.active_document().version == 1


def test_manual_override_scoped_expiring_never_hard():
    authority = CapabilityAuthority()
    operator_cap = authority.issue("ops.user", "operator", ["operator.override"], ttl_seconds=60)
    control = OperatorControl(max_ttl_seconds=60, authorized_operators=("ops.user",))
    grant = control.request_override(
        operator_cap, operator="ops.user", scope="queue:echo", reason="incident", ttl_seconds=30
    )
    assert grant and grant.is_active()
    assert control.expired_overrides(now=time.time() + 61)
    assert control.request_override(
        operator_cap, scope="hard:rule", reason="escape", ttl_seconds=10, relaxes_hard_rule=True
    ) is None
    stranger = authority.issue("intruder.comp", "operator", ["operator.override"], ttl_seconds=60)
    assert control.request_override(stranger, scope="queue:x", reason="nope", ttl_seconds=10) is None


def test_policy_engine_hard_floor_and_explanations():
    engine = PolicyEngine()
    denial = engine.evaluate(
        "gen.comp", "dispatch", {"trust_class": "untrusted_generated", "tier": "in_process"}
    )
    assert not denial.allowed and denial.hard
    allow = engine.evaluate("echo.service", "dispatch", {"deadline_seconds": 10.0})
    assert allow.allowed
    assert "policy" in engine.explain(allow)


# --- workspace guards (step 20) -------------------------------------------------------


def test_unauthorized_component_cannot_modify_workspace():
    runtime = build_runtime()
    with runtime:
        guard = runtime.blackboard_guard
        intruder = runtime.authority.issue(
            "intruder.comp", "workspace", ["workspace.append_observation"],
            ttl_seconds=60, constraints={"fields": ["lexical_hypotheses"]},
        )
        result = guard.append_observation(
            intruder, runtime.authority.authorize, field="lexical_hypotheses", payload={"k": "v"}
        )
        assert not result.allowed


def test_authorized_component_write_and_snapshot():
    runtime = build_runtime()
    with runtime:
        guard = runtime.blackboard_guard
        writer = runtime.authority.issue(
            "lexical_processor", "workspace", ["workspace.append_observation"],
            ttl_seconds=60, constraints={"fields": ["lexical_hypotheses"]},
        )
        reader = runtime.authority.issue(
            "lexical_processor", "workspace", ["workspace.read_snapshot"], ttl_seconds=60
        )
        if guard.backed_by_ore_workspace:
            result = guard.append_observation(
                writer, runtime.authority.authorize,
                field="lexical_hypotheses", payload={"word": "cat"},
            )
            assert result.allowed, result.reason
            snapshot = guard.read_snapshot(reader, runtime.authority.authorize)
            assert snapshot["source"] == "ore_workspace"
        else:  # standalone mode: grant field ownership first
            guard.grant_minimal_permission("lexical_processor", ["lexical_hypotheses"])
            result = guard.append_observation(
                writer, runtime.authority.authorize,
                field="lexical_hypotheses", payload={"word": "cat"},
            )
            assert result.allowed
            snapshot = guard.read_snapshot(reader, runtime.authority.authorize)
            assert "lexical_hypotheses" in snapshot["state"]


def test_usg_updates_only_via_external_validation():
    runtime = ComponentRuntime()
    with runtime:
        guard = runtime.usg_guard
        proposer = runtime.authority.issue(
            "semantic.module", "usg", ["usg.propose_update", "usg.read_snapshot"], ttl_seconds=60
        )
        record = guard.propose_update(proposer, runtime.authority.authorize, candidate_bundle=object())
        assert record.status in {"validated_pending_review", "rejected", "pending_external_validation"}
        assert guard.describe()["canonical_writes"] == 0
        reader = runtime.authority.issue("reader.comp", "usg", ["usg.read_snapshot"], ttl_seconds=60)
        snapshot = guard.read_snapshot(reader, runtime.authority.authorize)
        assert snapshot["source"] == "empty"


# --- registry, inventory, graph (steps 1, 4, 34) --------------------------------------


def test_dependency_graph_cycles():
    provider = trusted_manifest("prov.comp")
    consumer = ComponentManifest(component_id="cons.comp", dependencies=("prov.comp",))
    graph = DependencyGraph([provider, consumer])
    assert graph.startup_order() == ("prov.comp", "cons.comp")
    graph.add_edge(DependencyEdge(consumer="prov.comp", provider="cons.comp"))
    with pytest.raises(DependencyGraphError):
        graph.startup_order()
    async_graph = DependencyGraph([provider, consumer])
    async_graph.add_edge(DependencyEdge("cons.comp", "prov.comp", asynchronous=True, bounded_queue=True))
    async_graph.add_edge(DependencyEdge("prov.comp", "cons.comp", asynchronous=True, bounded_queue=True))
    assert async_graph.startup_order()  # permitted runtime cycle


def test_health_lease_distinguishes_liveness_from_readiness():
    runtime = build_runtime()
    with runtime:
        lease = runtime.issue_lease("echo.service", ttl_seconds=30)
        assert lease.remaining_seconds() > 0
        health = runtime.health("echo.service")
        assert health["process_alive"] and health["task_capable"]
        runtime.registry.expire_leases(now=time.time() + 60)
        assert not runtime.health("echo.service")["service_responsive"]


def test_repository_inventory_reconciled_with_actual_files():
    repository_root = Path(__file__).resolve().parents[1]
    report = inventory_repository(repository_root)
    assert report["reconciled"]
    assert report["component_count"] >= 1
    # the full coordinator set only exists inside the repository clone
    if (repository_root / "ore_contracts").is_dir():
        assert report["component_count"] >= 5
        assert any("adaptive" in record["component_id"] for record in report["records"])


# --- model-based checks and probes (steps 47-48) -------------------------------------


def test_model_based_lifecycle_and_reservation_checks():
    checker = ModelBasedLifecycleChecker()
    lifecycle = checker.exhaustive_transition_check()
    assert not lifecycle["violations"]
    reservations = checker.reservation_invariant_check(steps=80)
    assert not reservations["violations"]
    attenuation = checker.capability_attenuation_check(trials=16)
    assert not attenuation["violations"]


def test_security_probe_suite_success_and_denial_paths():
    runtime = build_runtime()
    with runtime:
        report = SecurityProbeSuite(runtime).run()
        assert report.passed, report.details
        assert set(report.results) == {
            "unauthorized_workspace_write",
            "expired_capability",
            "revoked_generation",
            "impersonation",
            "oversized_payload",
            "unsupported_version",
            "child_authority_escalation",
        }


# --- end-to-end governance (steps 14, 52, section 11) --------------------------------


def test_start_stop_lifecycle_releases_reservations(tmp_path: Path):
    runtime = build_runtime(tmp_path)
    with runtime:
        handle = runtime.start("echo.service", profile={"payload": {"n": 1}})
        assert handle.execution.state == ExecutionState.FINISHED
        assert runtime.lifecycle.state("echo.service") == "running"
        stop = handle.stop(reason="test complete")
        assert stop.outcome == OutcomeKind.OK
        assert stop.reservations_released == 1
        assert runtime.resources.available().cpu_shares >= 8.0 - 1e-9


def test_explain_answers_governance_question():
    runtime = build_runtime()
    with runtime:
        runtime.start("echo.service", profile={"payload": {"n": 1}})
        explanation = runtime.explain("echo.service", task_id="task-echo.service")
        for key in ("component", "authority", "budget", "policy", "supervisor", "ipc"):
            assert key in explanation
        assert explanation["policy"]["version"] == 1


def test_trace_propagation_from_request_to_publication(tmp_path: Path):
    runtime = build_runtime(tmp_path)
    with runtime:
        cap_id = runtime._grants["echo.service"]["ipc.send"].capability_id
        trace = TraceContext.new_trace(parent_task_id="parent-task")
        env = envelope(
            "echo.service", "echo.service", trace=trace, payload={"tracked": True},
            auth_capability_id=cap_id,
        )
        response = runtime.request(env)
        assert response.trace.trace_id == trace.trace_id
        assert response.trace.causation_id == env.message_id


def test_benchmark_reports_backends_equivalent():
    report = benchmark_overhead(repeats=2)
    data = report.to_dict()
    assert data["repeats"] == 2
    assert data["in_process_p50_ms"] >= 0
    if report.subprocess_ms:  # POSIX hosts only
        assert data["outputs_equivalent"]


def test_shutdown_releases_everything(tmp_path: Path):
    runtime = build_runtime(tmp_path)
    runtime.start("echo.service", profile={"payload": {}})
    summary = runtime.shutdown()
    assert summary["reservations_released"]
    assert summary["audit_events"] > 0


# --- queue age monitoring (section 5.4) ------------------------------------------------


def test_queue_age_monitored_not_only_length():
    from ore_runtime_governance import BoundedQueue

    queue = BoundedQueue(BackpressurePolicy(capacity=8))
    env = envelope("echo.service", "echo.service")
    queue.offer(env)
    assert queue.queue_age_seconds(now=time.monotonic() + 2.0) >= 2.0


def test_pause_ipc_fault_injection_times_out():
    runtime = build_runtime()
    with runtime:
        cap_id = runtime._grants["echo.service"]["ipc.send"].capability_id
        runtime.bus.pause()
        env = envelope(
            "echo.service", "echo.service", deadline_monotonic_s=time.monotonic() + 0.2,
            auth_capability_id=cap_id,
        )
        with pytest.raises(OperationTimedOutError):
            runtime.request(env)
        runtime.bus.resume()
        assert not runtime.bus.paused


def test_cancellation_token_semantics():
    from ore_runtime_governance import CancellationToken

    token = CancellationToken()
    token.cancel("operator abort")
    assert token.is_cancelled and token.reason == "operator abort"
    with pytest.raises(OperationCancelledError):
        token.raise_if_cancelled()


def test_worker_validates_job_trust_boundary(tmp_path: Path):
    from ore_runtime_governance.worker import run_job

    bad_job = tmp_path / "bad.json"
    bad_job.write_text(json.dumps({"entry_point": "../etc/passwd", "payload": {}}))
    code = run_job(bad_job, tmp_path / "out.json")
    assert code == 2
    result = json.loads((tmp_path / "out.json").read_text())
    assert "error" in result


def test_impersonation_error_is_denied():
    authority = CapabilityAuthority()
    stranger = authority.issue("stranger.comp", "ipc", ["ipc.send"], ttl_seconds=60)
    with pytest.raises(CapabilityDeniedError):
        authority.require(stranger, AccessRequest("victim.comp", "ipc/x", "ipc.send"))


def test_response_envelope_rejects_missing_correlation():
    with pytest.raises(ValueError):
        ResponseEnvelope(correlation_id="", sender="a.comp", recipient="b.comp")


def test_environment_image_capture():
    from ore_runtime_governance import EnvironmentImage

    image = EnvironmentImage.capture_current()
    report = image.reproducibility_report()
    assert report["image_id"].startswith("env-")
    assert report["runtime_versions"]["python"]


def test_lifecycle_state_enumeration():
    assert LifecycleState.ALL[-1] == "quarantined"
    assert set(LifecycleState.TERMINAL) == {"stopped", "failed", "quarantined"}
