"""Selective CRDT replication with semantic conflict preservation (Phase D)."""
from ore_distributed.replication.crdts import (
    CRDTSpec,
    GSet,
    LWWRegister,
    ORSet,
    PNCounter,
    check_merge_laws,
)
from ore_distributed.replication.replica import (
    CrdtReplica,
    DistributedObservation,
    ObservationStatus,
)
from ore_distributed.replication.conflicts import SemanticConflictRecord
from ore_distributed.replication.reconciliation import reconcile_replicas
from ore_distributed.replication.tombstones import TombstoneCollector
from ore_distributed.replication.promotion import USGPromotionGate

__all__ = [
    "CRDTSpec",
    "GSet",
    "LWWRegister",
    "ORSet",
    "PNCounter",
    "check_merge_laws",
    "CrdtReplica",
    "DistributedObservation",
    "ObservationStatus",
    "SemanticConflictRecord",
    "reconcile_replicas",
    "TombstoneCollector",
    "USGPromotionGate",
]
