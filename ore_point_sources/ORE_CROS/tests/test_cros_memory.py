"""Acceptance tests for intelligent memory management (Steps 27-35)."""
import pytest

from ore_cros.policies.memory import (
    AcceleratorMemoryManager, CLOCKReplacer, CompressionTierPolicy,
    CxlExtensionPoint, DamonAdapter, DataClassProfile, LRUReplacer,
    MultiGenerationReplacer, NumaTopology, NumaNode, TieredCache,
    WorkingSetEstimator,
)


def test_working_set_estimates_respond_to_access_patterns():
    """Acceptance: working-set estimates respond to changing access patterns."""
    estimator = WorkingSetEstimator(short_window=1.0, medium_window=10.0)
    # Phase 1: hot loop over two objects.
    for tick in range(10):
        estimator.observe("task", "a", now=float(tick))
        estimator.observe("task", "b", now=float(tick))
    early = estimator.estimate("task", now=9.0)
    assert early.short_window_size == 2
    # Phase 2: the working set grows to five new objects at t=10.
    for tick in range(10, 20):
        for letter in "cdefg":
            estimator.observe("task", letter, now=float(tick))
    transition = estimator.estimate("task", now=10.0)
    # The trailing window still sees the old set plus the new one.
    assert transition.short_window_size == 7
    assert transition.rate_of_change > 0
    grown = estimator.estimate("task", now=20.0)
    assert grown.short_window_size == 5
    assert grown.sustained is True
    # Phase 3: quiet period shrinks the short-window estimate.
    quiet = estimator.estimate("task", now=30.0)
    assert quiet.short_window_size == 0
    assert estimator.prune(now=30.0) >= 0


def test_cache_tiering_preserves_values_and_provenance():
    """Acceptance: cache tiering never changes semantic retrieval results
    or provenance (Step 28)."""
    cache = TieredCache(hot=2, warm=2, cold=4)
    cache.put("k1", {"finding": "alpha"}, provenance="ore_semantic/v2")
    assert cache.get("k1") == {"finding": "alpha"}
    # Promote through repeated access.
    for _ in range(4):
        assert cache.get("k1") == {"finding": "alpha"}
    assert cache.provenance_of("k1") == "ore_semantic/v2"
    assert cache.tier_of("k1") == "hot"
    assert cache.tier_moves()
    assert 0.0 <= cache.hit_rate() <= 1.0


def test_tier_eviction_only_from_cold_tier():
    """Eviction drains the cold tier; hotter tiers demote instead (Step 28)."""
    cache = TieredCache(hot=1, warm=1, cold=2)
    for index in range(6):
        cache.put(f"key{index}", index)
    sizes = cache.tier_sizes()
    assert sizes["hot"] <= 1 and sizes["warm"] <= 1
    assert sum(sizes.values()) <= 4


def test_lru_is_scan_sensitive_clock_resists():
    """LRU/CLOCK baselines are compared with measurable hit rates (Step 29)."""
    working_set = [f"w{index}" for index in range(4)]
    scan = [f"s{index}" for index in range(50)]

    lru = LRUReplacer(capacity=8)
    for key in working_set * 3:
        lru.touch(key, inserted=True)
    for key in scan:
        lru.touch(key, inserted=True)  # one big scan evicts everything
    for key in working_set:
        lru.touch(key)
    lru_hit_rate = lru.hit_rate()

    clock = CLOCKReplacer(capacity=8)
    for key in working_set * 3:
        clock.touch(key, inserted=True)
    for key in scan:
        clock.touch(key, inserted=True)
    for key in working_set:
        clock.touch(key)
    clock_hit_rate = clock.hit_rate()

    # CLOCK's reference bits give the warm set a second chance.
    assert clock_hit_rate >= lru_hit_rate


def test_multi_generation_replacer_evicts_oldest_first():
    """The MGLRU-inspired replacer drains the oldest generation (Step 30)."""
    replacer = MultiGenerationReplacer(capacity=4, generations=4, aging_interval=4)
    for key in "abcd":
        replacer.touch(key, inserted=True)
    # Four insertions age the first generation into the second ring.
    assert replacer.generation_sizes()[0] == 0
    assert replacer.generation_sizes()[1] == 4
    # Touch 'a' again: it is promoted into the youngest generation.
    replacer.touch("a")
    assert replacer.generation_sizes()[0] == 1
    evicted = replacer.evict()
    assert evicted is not None and evicted != "a"
    assert sum(replacer.generation_sizes()) <= 4


def test_damon_adapter_reads_without_mutating():
    """DAMON telemetry stays read-only and reports unavailability (Step 31)."""
    adapter = DamonAdapter(damon_root="/nonexistent-damon-path")
    assert adapter.available() is False
    reading = adapter.read_access_pattern()
    assert reading.available is False
    assert "not present" in reading.note


def test_numa_locality_only_when_topology_available():
    """Acceptance: NUMA decisions account for locality only when topology
    is available (Step 32)."""
    empty = NumaTopology([])
    assert empty.locality_available is False
    assert empty.choose_node([0]) is None
    with pytest.raises(RuntimeError):
        empty.placement_cost(0, [0])

    topology = NumaTopology([
        NumaNode(0, (0, 1), 8192.0), NumaNode(1, (2, 3), 8192.0),
    ])
    assert topology.locality_available is True
    assert topology.placement_cost(0, [0]) == pytest.approx(0.0)
    assert topology.placement_cost(0, [1]) > 0.0
    assert topology.choose_node([1]) == 1
    assert topology.choose_node([0, 1]) in (0, 1)


def test_numa_discovery_from_fake_sysfs(tmp_path):
    """NUMA discovery parses sysfs node directories when present."""
    node_dir = tmp_path / "node0"
    node_dir.mkdir()
    (node_dir / "cpulist").write_text("0-3\n")
    (node_dir / "meminfo").write_text("MemTotal:       8388608 kB\n")
    topology = NumaTopology.discover(root=tmp_path)
    assert len(topology.nodes) == 1
    assert topology.nodes[0].cpus == (0, 1, 2, 3)
    assert topology.nodes[0].memory_mib == pytest.approx(8192.0)
    assert topology.locality_available is False  # single node: no locality


def test_compression_tiering_cost_model():
    """Compression and tiering follow access frequency and cost (Step 33)."""
    policy = CompressionTierPolicy(compression_ratio=0.5)
    hot = DataClassProfile(
        name="active-findings", access_frequency=10.0,
        decompression_cost=0.5, latency_tolerance=0.1,
    )
    cold = DataClassProfile(
        name="archive", access_frequency=0.0,
        decompression_cost=0.5, latency_tolerance=1.0,
    )
    tolerant = DataClassProfile(
        name="bulk-index", access_frequency=5.0,
        decompression_cost=0.05, latency_tolerance=1.0,
    )
    assert policy.recommend(hot).decision == "in-memory"
    assert policy.recommend(cold).decision == "ssd-tier"
    assert policy.recommend(tolerant).decision == "compressed"


def test_gpu_admission_rejects_or_defers_oversized_tasks():
    """Acceptance: GPU tasks are rejected/deferred when accelerator memory
    is insufficient (Step 34)."""
    manager = AcceleratorMemoryManager(capacity_mib=4096.0, transfer_batch_size=64.0)
    verdict, _ = manager.admit("t1", working_set_mib=1024.0)
    assert verdict == "ADMIT"
    # 3500 MiB exceeds the 3072 MiB headroom but not the 4096 MiB capacity.
    verdict, reason = manager.admit("t2", working_set_mib=3500.0)
    assert verdict == "DEFER" and "headroom" in reason
    verdict, reason = manager.admit("t3", working_set_mib=8192.0)
    assert verdict == "REJECT" and "capacity" in reason
    plan = manager.plan_transfer("t1", 150.0)
    assert plan["batch_count"] == 3
    manager.release("t1")
    assert manager.headroom_mib() == pytest.approx(4096.0)


def test_cxl_extension_point_stays_minimal():
    """CXL remains a future extension interface without premature code (Step 35)."""
    extension = CxlExtensionPoint()
    assert extension.devices() == {}
    assert extension.placement_cost("cxl-0") is None
    extension.register_device("cxl-0", {"capacity_mib": 65536.0, "access_cost": 1.2})
    assert extension.placement_cost("cxl-0") == pytest.approx(1.2)
