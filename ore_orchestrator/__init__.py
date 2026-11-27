"""Unified public entry point for the ORE Neurocomputational repository."""
from .catalog import OREPackageCatalog, PackageDescriptor
from .orchestrator import ComponentOutcome, OREExecutionReport, OREOrchestrator

__version__ = "0.1.0"

__all__ = [
    "ComponentOutcome",
    "OREExecutionReport",
    "OREOrchestrator",
    "OREPackageCatalog",
    "PackageDescriptor",
    "__version__",
]
