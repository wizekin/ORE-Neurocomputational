"""Explicit USG concept mapping (specification Step 33).

Validated event labels are translated into USG identifiers through explicit mappings
only; when no mapping exists, an explicit unknown/uncategorized identifier is published
rather than an invented class.
"""
from __future__ import annotations

from typing import Any

__all__ = ("UNKNOWN_USG_ID", "USGConceptMapper")

UNKNOWN_USG_ID = "usg:temporal:unknown@1"


class USGConceptMapper:
    """Map temporal event labels to USG identifiers with an explicit, versioned table."""

    #: The explicit label -> USG identifier mapping table (never invented at runtime).
    DEFAULT_MAPPINGS: dict[str, str] = {
        "change_point": "usg:temporal:change_point@1",
        "level_shift": "usg:temporal:level_shift@1",
        "anomaly_spike": "usg:temporal:anomaly_spike@1",
        "distribution_shift": "usg:temporal:distribution_shift@1",
        "sensor_fault": "usg:temporal:sensor_fault@1",
        "sampling_gap": "usg:temporal:sampling_gap@1",
        "spike_burst": "usg:temporal:spike_burst@1",
        "forecast": "usg:temporal:forecast@1",
        "state_summary": "usg:temporal:state_summary@1",
    }

    def __init__(self, mappings: dict[str, str] | None = None) -> None:
        """Configure the mapper with the default or a caller-supplied explicit table."""
        self._mappings = dict(mappings) if mappings else dict(self.DEFAULT_MAPPINGS)

    def register_mapping(self, label: str, usg_id: str) -> None:
        """Register a new explicit mapping; identifiers must look like USG ids."""
        if not label or not isinstance(label, str):
            raise ValueError("label must be a non-empty string")
        if not usg_id.startswith("usg:") or "@" not in usg_id:
            raise ValueError("usg_id must follow the 'usg:<scope>:<record>@<version>' format")
        self._mappings[label] = usg_id

    def map(self, label: str) -> tuple[str, bool]:
        """Return (usg_id, mapped); unmapped labels yield the explicit unknown id."""
        usg_id = self._mappings.get(label)
        if usg_id is None:
            return (UNKNOWN_USG_ID, False)
        return (usg_id, True)

    def map_event_type(self, event_type: str) -> tuple[str, bool]:
        """Map a typed event record's event_type label."""
        return self.map(event_type)

    def mappings(self) -> dict[str, str]:
        """Return the explicit mapping table for provenance."""
        return dict(self._mappings)

    def describe(self) -> dict[str, Any]:
        """Describe the mapper for diagnostics."""
        return {"mappings": self.mappings(), "unknown_id": UNKNOWN_USG_ID,
                "policy": "explicit mappings only; unknown labels map to the explicit "
                          "unknown/uncategorized identifier"}
