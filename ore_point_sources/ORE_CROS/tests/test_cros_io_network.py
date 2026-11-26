"""Acceptance tests for I/O, networking and storage policies (Steps 42-46)."""
import pytest

from ore_cros.actions import ActionType, PolicyOrigin, ResourceAction
from ore_cros.policies.io_network import (
    AsyncIoBackend, DurabilityContract, DurabilityContractRegistry,
    IoNetworkProfiler, SpecializedNetworkGate, StorageCapabilityMatrix,
)
from ore_cros.resource_model import ResourceKind, machine_scope
from ore_cros.telemetry.schemas import Measurement, MetricKind

NOW = 1_700_000_000.0


def _measurement(resource, metric, value, unit=None):
    """Build one machine-scope measurement for I/O tests."""
    from ore_cros.telemetry.schemas import canonical_unit

    return Measurement(
        resource=resource, metric=metric, value=value,
        unit=unit or canonical_unit(resource, metric),
        timestamp=NOW, scope=machine_scope(), source="test",
    )


def test_io_profiler_identifies_bottlenecks_with_evidence():
    """Step 42: I/O and network demand profiling identifies real bottlenecks."""
    measurements = [
        _measurement(ResourceKind.IO, MetricKind.UTILIZATION, 0.95),
        _measurement(ResourceKind.NET, MetricKind.UTILIZATION, 0.40),
        _measurement(ResourceKind.IO, MetricKind.LATENCY_SECONDS, 0.080, unit="s"),
    ]
    profile = IoNetworkProfiler.profile(measurements)
    assert profile.bottleneck == "IO"
    assert profile.evidence
    assert profile.p99_latency_seconds == pytest.approx(0.080)


def test_async_io_backend_falls_back_cleanly():
    """Acceptance: unsupported io_uring/platform interfaces fall back cleanly
    (Step 43)."""
    backend = AsyncIoBackend(measured_gain=0.5)
    name, rationale = backend.select_backend(io_uring_supported=False)
    assert name == AsyncIoBackend.PORTABLE_FALLBACK
    assert "unsupported" in rationale
    name, rationale = backend.select_backend(io_uring_supported=True)
    assert name == AsyncIoBackend.IO_URING
    unbenchmarked = AsyncIoBackend(measured_gain=None)
    name, rationale = unbenchmarked.select_backend(io_uring_supported=True)
    assert name == AsyncIoBackend.PORTABLE_FALLBACK
    assert "benchmark evidence" in rationale
    marginal = AsyncIoBackend(measured_gain=0.02)
    name, rationale = marginal.select_backend(io_uring_supported=True)
    assert name == AsyncIoBackend.PORTABLE_FALLBACK


def test_specialized_network_gate_requires_evidence():
    """eBPF/XDP/DPDK/Demikernel stay gated behind explicit evidence (Step 44)."""
    gate = SpecializedNetworkGate()
    authorized, rationale = gate.authorize("eBPF")
    assert not authorized and "missing" in rationale
    evidenced = SpecializedNetworkGate(
        evidence={"workload_requirement": True, "expertise": True, "benchmark": True},
    )
    authorized, rationale = evidenced.authorize("XDP")
    assert authorized
    unknown, _ = evidenced.authorize("raw-sockets")
    assert not unknown


def test_storage_matrix_returns_candidates_not_defaults():
    """Step 45: storage selection is a capability matrix, not one default fs."""
    matrix = StorageCapabilityMatrix()
    snapshot_candidates = matrix.recommend(["snapshots"])
    assert any(item["filesystem"] == "Btrfs" for item in snapshot_candidates)
    integrity = matrix.recommend(["integrity", "replication"])
    assert any("ZFS" in item["filesystem"] for item in integrity)
    assert matrix.recommend([]) == ()


def test_durability_contracts_block_silent_downgrades():
    """Acceptance: resource optimization cannot violate storage durability
    contracts (Step 46)."""
    registry = DurabilityContractRegistry([
        DurabilityContract(
            data_class="workspace-state", durability="fsync", consistency="atomic",
            backup=True, recovery_point_objective_seconds=60.0,
            recovery_time_objective_seconds=300.0,
        ),
    ])
    weaker = DurabilityContract(
        data_class="workspace-state", durability="none", consistency="best-effort",
        backup=False, recovery_point_objective_seconds=0.0,
        recovery_time_objective_seconds=0.0,
    )
    allowed, reason = registry.validate_transition("workspace-state", weaker)
    assert not allowed and "cannot downgrade" in reason
    no_backup = DurabilityContract(
        data_class="workspace-state", durability="fsync", consistency="atomic",
        backup=False, recovery_point_objective_seconds=60.0,
        recovery_time_objective_seconds=300.0,
    )
    allowed, reason = registry.validate_transition("workspace-state", no_backup)
    assert not allowed and "backup cannot be removed" in reason
    action = ResourceAction(
        action_id="tier-change", action_type=ActionType.CHANGE_CACHE_TIER,
        parameters={
            "tier_sizes": {"hot": 8, "warm": 16, "cold": 64},
            "affected_data_classes": ["workspace-state"],
        },
        rationale="shrink hot tier under memory pressure",
        policy_origin=PolicyOrigin.DETERMINISTIC, policy_version="v1",
        state_version=1,
    )
    ok, reason = registry.validate_action(action)
    assert not ok


def test_durability_contract_validation():
    """Durability contracts reject malformed declarations."""
    with pytest.raises(ValueError):
        DurabilityContract(
            data_class="x", durability="turbo", consistency="atomic",
            backup=True, recovery_point_objective_seconds=1.0,
            recovery_time_objective_seconds=1.0,
        )
    with pytest.raises(ValueError):
        DurabilityContract(
            data_class="x", durability="fsync", consistency="atomic",
            backup=True, recovery_point_objective_seconds=-1.0,
            recovery_time_objective_seconds=1.0,
        )
    with pytest.raises(ValueError):
        DurabilityContractRegistry([
            DurabilityContract(
                data_class="x", durability="fsync", consistency="atomic",
                backup=True, recovery_point_objective_seconds=1.0,
                recovery_time_objective_seconds=1.0,
            ),
            DurabilityContract(
                data_class="x", durability="replicated", consistency="strict",
                backup=True, recovery_point_objective_seconds=1.0,
                recovery_time_objective_seconds=1.0,
            ),
        ])
