"""Source-ordered operation registry for the 48 steps of Point 17.

Maps every algorithmic step of the distributed coordination algorithm to its
implementation module, phase, protocol rule, and architectural reference,
following the repository's registry convention.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

#: Version of the Point 17 implementation.
ORE_DISTRIBUTED_VERSION = "0.1.0"

#: Contract identifier for the distributed coordination layer.
ORE_DISTRIBUTED_CONTRACT = "ORE.DISTRIBUTED.COORDINATION/0.1.0"


@dataclass(frozen=True, slots=True)
class OperationSpec:
    """Describe one Point 17 operation and its implementation mapping."""

    operation_id: str
    phase: str
    source_step: int
    name: str
    purpose: str
    protocol_rule: str
    implementation_module: str
    evidence_level: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Return the operation record as a plain dictionary."""
        return asdict(self)


_ROWS: tuple[tuple[int, str, str, str, str, str, tuple[str, ...], tuple[str, ...]], ...] = (
    # Phase A - deployment classification and state boundaries
    (1, "A", "Detect deployment topology", "Classify single process, local processes, or cluster from a topology descriptor.", "mode in {single_process, multi_process_local, multi_machine_cluster}", "ore_distributed.core.deployment", ("topology_descriptor",), ("deployment_mode",)),
    (2, "A", "Classify every state object", "Assign each state schema one of four classes; reject unclassified schemas.", "class(schema) in {authoritative, mergeable, immutable_event, agent_local}", "ore_distributed.core.state_class", ("state_schemas",), ("classifications",)),
    (3, "A", "Define consistency requirements", "Select linearizable, eventual, causal, read-your-writes, or local-only per record.", "consistency(class) in allowed(class)", "ore_distributed.core.state_class", ("classifications",), ("consistency_requirements",)),
    (4, "A", "Define failure and threat models", "Separate crash, partition, delay, duplication, reorder, drift, and Byzantine failures.", "threat.byzantine -> BFT else Raft", "ore_distributed.core.state_class", ("deployment_threat_model",), ("protocol_family",)),
    (5, "A", "Establish protocol and schema compatibility", "Version, authenticate, and cluster-bind every message before state mutation.", "envelope carries protocol_version, schema_version, cluster_id, sender, type", "ore_distributed.core.protocol", ("messages", "credentials"), ("verified_envelopes",)),
    (6, "A", "Create stable node identities", "Provision durable identities and credentials bound to authorization and audit.", "identity != f(IP, hostname, container)", "ore_distributed.core.identity", ("node_registration",), ("node_identities",)),
    # Phase B - membership, discovery, secure communication
    (7, "B", "Maintain a node registry", "Track identity, endpoints, capabilities, health, and authorization per node.", "health in {discovered, authenticated, eligible, voting, draining, removed}", "ore_distributed.membership.registry", ("membership_events",), ("node_registry",)),
    (8, "B", "Authenticate membership changes", "Require authenticated admission and authorized removal with full audit records.", "change(initiator, reason, config_version, committing_event)", "ore_distributed.membership.admission", ("membership_change_requests",), ("membership_changes", "audit_trail")),
    (9, "B", "Exchange capability descriptors", "Advertise modules, USG schema versions, hardware, and workload classes as claims.", "capabilities are claims requiring validation", "ore_distributed.membership.registry", ("capability_messages",), ("capability_claims",)),
    (10, "B", "Establish authenticated transport", "Mutually authenticate and authorize every message class with bounded sizes.", "transport auth separate from semantic trust", "ore_distributed.membership.transport", ("credentials", "endpoints"), ("authenticated_channels",)),
    (11, "B", "Implement bounded retries and backpressure", "Exponential backoff with jitter and bounded queues and retry budgets.", "delay(attempt) = min(max, base*2^attempt) +/- jitter", "ore_distributed.membership.transport", ("transient_failures",), ("retry_schedules", "backpressure_events")),
    (12, "B", "Separate discovery from consensus membership", "Only the consensus membership protocol changes the voter configuration.", "discovered != voter without authorized change", "ore_distributed.membership.admission", ("discovered_nodes", "voting_config"), ("voter_configuration",)),
    # Phase C - authoritative coordination with Raft
    (13, "C", "Initialize the replicated state machine", "Define the deterministic state machine for authoritative operational records.", "apply(entry) is deterministic and versioned", "ore_distributed.consensus.state_machine", ("client_commands",), ("authoritative_state",)),
    (14, "C", "Implement leader election", "Randomized timeouts, term increment, majority election, split-vote avoidance.", "votes >= Q = floor(N/2)+1", "ore_distributed.consensus.raft", ("election_timeouts",), ("leader", "term")),
    (15, "C", "Enforce term and vote safety", "Persist term and vote before responding; one vote per term; log freshness check.", "grant iff voted_for in {null, candidate} and log_fresh", "ore_distributed.consensus.raft", ("request_vote_args",), ("vote_replies",)),
    (16, "C", "Replicate commands through the leader", "AppendEntries with prev-log check, truncation of uncommitted suffixes, match/next indices.", "match(i) iff log[i].term agrees on prefix", "ore_distributed.consensus.raft", ("client_commands",), ("replicated_log", "match_indices")),
    (17, "C", "Commit entries using protocol rules", "Advance commit only with current-term replication to a quorum; acknowledge after durable commit.", "commit = replicated majority AND entry.term == current_term", "ore_distributed.consensus.raft", ("replication_progress",), ("commit_index", "client_ack")),
    (18, "C", "Apply committed entries deterministically", "Apply in log order with command IDs and deduplication records.", "apply(command_id) idempotent", "ore_distributed.consensus.state_machine", ("committed_entries",), ("applied_state", "dedup_records")),
    (19, "C", "Handle leader failure and recovery", "New election reconstructs progress; an old leader steps down on higher term.", "term(observed) > current_term -> step down", "ore_distributed.consensus.raft", ("leader_loss", "higher_term_messages"), ("new_leader", "step_down_event")),
    (20, "C", "Implement safe membership changes", "Validate quorum availability, serialize changes, record configuration epochs.", "one change in flight; quorum before and after", "ore_distributed.consensus.reconfiguration", ("membership_change_requests",), ("config_epochs", "voter_sets")),
    (21, "C", "Use leases only with explicit clock assumptions", "Define drift bounds, renewal rules, and fencing tokens; epoch checks at the resource.", "write(resource) requires fencing_epoch >= issued", "ore_distributed.consensus.leases", ("lease_requests", "clock_drift_bounds"), ("lease_grants", "fencing_rejections")),
    (22, "C", "Keep consensus scope narrow", "Replicate only authoritative control-plane records; keep observations off the log.", "command_type in control_plane_types", "ore_distributed.consensus.state_machine", ("proposed_commands",), ("accepted_commands",)),
    # Phase D - selective CRDT replication
    (23, "D", "Identify merge-safe data types", "Select CRDTs only where concurrency semantics are well defined.", "kind in {gset, orset, pncounter, lww_register} with documented assumptions", "ore_distributed.replication.crdts", ("state_schemas",), ("crdt_kinds",)),
    (24, "D", "Define a merge algebra for each CRDT", "Specify state space, partial order, and merge; property-test the laws.", "merge associative, commutative, idempotent", "ore_distributed.replication.crdts", ("generated_states",), ("merge_law_reports",)),
    (25, "D", "Distinguish duplicate identity from duplicate content", "Stable event IDs and origins; dedupe by ID, never collapse matching text.", "same text, different origin -> two observations", "ore_distributed.replication.replica", ("delivered_events",), ("deduplicated_events",)),
    (26, "D", "Preserve semantic contradictions", "Merge evidence records without declaring either claim true; create conflict objects.", "conflict links claims, contexts, sources, timestamps", "ore_distributed.replication.conflicts", ("conflicting_claims",), ("conflict_records", "usg_routing")),
    (27, "D", "Apply CRDT updates locally and asynchronously", "Record locally with unique identities and queue dissemination; statuses stay separate.", "local acceptance != global visibility != semantic verification", "ore_distributed.replication.replica", ("local_updates",), ("replica_state", "dissemination_queue")),
    (28, "D", "Reconcile replicas", "Exchange state per the delivery model; converge with causal metadata; repeated reconciliation is a no-op.", "converged(a,b) -> reconcile(a,b) is identity", "ore_distributed.replication.reconciliation", ("replica_states",), ("merged_replicas", "convergence_reports")),
    (29, "D", "Bound tombstones and history growth", "Collect deletion metadata only after causal stability; never merely because it is old.", "collect(tag) iff all known replicas observed tag", "ore_distributed.replication.tombstones", ("tombstones", "replica_observations"), ("collected_tombstones", "resurrection_blocks")),
    (30, "D", "Gate authoritative semantic promotion", "Require schema, provenance, confidence, context, and contradiction checks before USG assertion.", "consensus orders approval; convergence is not truth", "ore_distributed.replication.promotion", ("candidate_assertions", "open_conflicts"), ("promotion_outcomes",)),
    # Phase E - gossip, event ordering, provenance
    (31, "E", "Build a peer sampling strategy", "Bounded randomized peers with failure-domain diversity; no static hub.", "fanout f <= max_peers with per-domain cap", "ore_distributed.gossip.peer_sampling", ("authenticated_members",), ("peer_views",)),
    (32, "E", "Disseminate updates using gossip", "Digest comparison then selective repair with rate limits and payload caps.", "digest-compare -> repair missing; batch <= cap", "ore_distributed.gossip.dissemination", ("local_entries", "peer_digests"), ("repaired_entries", "convergence")),
    (33, "E", "Add failure suspicion", "Suspicion states before removal; never declare failure from one missed heartbeat.", "unreachable -> suspected -> removed with timeouts", "ore_distributed.gossip.suspicion", ("heartbeats", "missed_heartbeats"), ("suspicion_states",)),
    (34, "E", "Update Hybrid Logical Clocks", "Merge physical and logical components monotonically; persist across restarts.", "hlc = (p, l, node); merge keeps monotone order", "ore_distributed.time.hlc", ("local_events", "received_timestamps"), ("hlc_timestamps",)),
    (35, "E", "Record explicit causal dependencies", "Parent event IDs, dependency versions, and vector clocks for causality.", "hlc order key != causality; parents define it", "ore_distributed.time.causal", ("event_histories",), ("causal_metadata", "concurrency_classification")),
    (36, "E", "Create immutable provenance envelopes", "Event ID, actor, action, digest, sources, parents, HLC, protocol, signature.", "envelope fields are immutable after publication", "ore_distributed.provenance.envelope", ("published_events",), ("provenance_envelopes",)),
    (37, "E", "Verify event integrity and lineage", "Authenticate, digest-check, schema-check, and quarantine missing dependencies.", "reject malformed; quarantine missing parents", "ore_distributed.provenance.journal", ("received_envelopes",), ("audit_trail", "quarantine", "rejections")),
    # Phase F - placement, execution, recovery
    (38, "F", "Introduce consistent hashing only when needed", "Hash ring with virtual nodes, replication factor, and incremental rebalancing.", "owners(key) = next r distinct nodes on ring", "ore_distributed.placement.consistent_hashing", ("keys", "ring_members"), ("placements", "rebalance_plans")),
    (39, "F", "Route data by semantic and operational ownership", "Distinct partition keys per data class; control metadata in consensus.", "partition_key = namespace(class):item", "ore_distributed.placement.router", ("data_items", "usg_queries"), ("shard_placements",)),
    (40, "F", "Schedule distributed tasks with leases and fencing", "Assign with lease expiry, attempt, and fencing epoch; validate ownership on commit.", "reassign only after state machine advances ownership", "ore_distributed.execution.tasks", ("task_assignments", "worker_outputs"), ("task_records", "stale_rejections")),
    (41, "F", "Make task execution retry-safe", "Idempotency keys, deterministic outputs, outbox/inbox for external effects.", "at-least-once delivery; effects exactly-once via idempotency", "ore_distributed.execution.idempotency", ("task_deliveries", "external_effects"), ("effect_records", "dedup_stats")),
    (42, "F", "Checkpoint coordinated workflows", "Persist state, committed task IDs, digests, dependency graph, epoch, cursor.", "checkpoint valid iff artifacts and provenance durable", "ore_distributed.execution.checkpoints", ("workflow_states", "artifacts"), ("checkpoints",)),
    (43, "F", "Recover after node or cluster disruption", "Rejoin, reconcile consensus, restore checkpoints, exchange CRDT summaries, replay events.", "recovered state published only after reconciliation", "ore_distributed.recovery.recovery", ("restart_events", "durable_state"), ("recovery_reports",)),
    (44, "F", "Handle partitions explicitly", "Majority continues; minority blocked; provisional CRDT visibility; degraded status exposed.", "minority cannot commit authoritative writes", "ore_distributed.recovery.partitions", ("partition_events",), ("degraded_status", "provisional_visibility")),
    # Phase G - ORE/USG/CROS integration and observability
    (45, "G", "Add a distributed coordination adapter", "Package boundary with interfaces for every subsystem behind adapters.", "interfaces over implementations; in-memory local adapters", "ore_distributed.adapters", ("deployment_modes",), ("adapter_interfaces",)),
    (46, "G", "Connect the shared cognitive workspace", "Typed workspace events route to authoritative, mergeable, or local paths.", "workspace publishes events, not replicated memory", "ore_distributed.integration.workspace", ("workspace_events",), ("routed_events",)),
    (47, "G", "Integrate USG and CROS", "USG validates semantics before promotion; CROS adapts resources within safe bounds.", "CROS cannot change consensus safety parameters", "ore_distributed.integration.usg_adapter + ore_distributed.integration.cros_adapter", ("candidate_assertions", "resource_telemetry"), ("validation_outcomes", "bounded_adaptations")),
    (48, "G", "Instrument, audit, and degrade safely", "Measure the Step 48 metric set; provide local-only and degraded modes.", "quorum unavailable -> read-only degraded mode", "ore_distributed.integration.observability", ("protocol_events",), ("metrics", "degraded_status")),
)


def _build_registry() -> tuple[OperationSpec, ...]:
    """Build the immutable 48-operation registry."""
    specs = []
    for (
        step,
        phase,
        name,
        purpose,
        rule,
        module,
        inputs,
        outputs,
    ) in _ROWS:
        specs.append(
            OperationSpec(
                operation_id=f"ore.dist.p17.step{step:02d}",
                phase=f"Phase {phase}",
                source_step=step,
                name=name,
                purpose=purpose,
                protocol_rule=rule,
                implementation_module=module,
                evidence_level="distributed_systems_protocol_guarantee; behavioral tests cover the acceptance set",
                inputs=inputs,
                outputs=outputs,
            )
        )
    return tuple(specs)


#: The canonical 48-operation registry for Point 17.
OPERATION_REGISTRY: tuple[OperationSpec, ...] = _build_registry()


def registry_as_dicts() -> list[dict[str, object]]:
    """Return the registry as plain dictionaries for serialization."""
    return [spec.to_dict() for spec in OPERATION_REGISTRY]


def operation_count() -> int:
    """Return the number of registered operations (must be 48)."""
    return len(OPERATION_REGISTRY)


def phases() -> tuple[str, ...]:
    """Return the ordered phase names of the algorithm."""
    ordered: list[str] = []
    for spec in OPERATION_REGISTRY:
        if spec.phase not in ordered:
            ordered.append(spec.phase)
    return tuple(ordered)


__all__ = [
    "OperationSpec",
    "OPERATION_REGISTRY",
    "registry_as_dicts",
    "operation_count",
    "phases",
    "ORE_DISTRIBUTED_VERSION",
    "ORE_DISTRIBUTED_CONTRACT",
]
