"""Safety layer: hard constraints, criticality, watchdogs and fallback (Steps 13, 47-48, 52)."""
from __future__ import annotations

from .constraints import (
    CriticalityMatrix, HardConstraintSet, ConstraintVerdict, QuotaLedger,
    RealTimeClaimError, RealTimeTaskSpec, WcetAnalysis, WcetAnalyzer,
)
from .fallback import (
    HysteresisController, OscillationDetector, PidController, RollbackManager,
    SafeFallbackPolicy, Watchdog, WatchdogVerdict,
)

__all__ = [
    "CriticalityMatrix", "HardConstraintSet", "ConstraintVerdict", "QuotaLedger",
    "RealTimeClaimError", "RealTimeTaskSpec", "WcetAnalysis", "WcetAnalyzer",
    "HysteresisController", "OscillationDetector", "PidController",
    "RollbackManager", "SafeFallbackPolicy", "Watchdog", "WatchdogVerdict",
]
