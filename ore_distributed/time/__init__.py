"""Hybrid Logical Clocks and causal dependency metadata (Steps 34-35)."""
from ore_distributed.time.hlc import HybridLogicalClock, HLCTimestamp, monotonic_restart_state
from ore_distributed.time.causal import (
    CausalHistory,
    VectorClock,
    events_concurrent,
    happens_before,
)

__all__ = [
    "HybridLogicalClock",
    "HLCTimestamp",
    "monotonic_restart_state",
    "CausalHistory",
    "VectorClock",
    "happens_before",
    "events_concurrent",
]
