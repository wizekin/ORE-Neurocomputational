"""Shared lifecycle and source-rights policy for read-only USG adapters."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

ACCEPTED_USG_STATES = frozenset({"ACCEPTED", "VALIDATED_FOR_PROFILE"})
CANDIDATE_USG_STATES = frozenset({"CANDIDATE"})
LOCAL_IMPORT_RIGHTS = frozenset({"KNOWN_PERMITTED", "INTERNAL"})


@dataclass(frozen=True, slots=True)
class USGImportPolicy:
    """One explicit policy shared by lexical and semantic import adapters."""

    include_candidate_states: bool = False
    allowed_local_rights: frozenset[str] = LOCAL_IMPORT_RIGHTS

    def __post_init__(self) -> None:
        """Reject ambiguous policy configuration."""
        if not isinstance(self.include_candidate_states, bool):
            raise TypeError("include_candidate_states must be a bool")
        if not self.allowed_local_rights or any(
            item not in {"KNOWN_PERMITTED", "INTERNAL", "RESTRICTED", "UNKNOWN", "PROHIBITED"}
            for item in self.allowed_local_rights
        ):
            raise ValueError("allowed_local_rights contains an unsupported rights state")

    def state_enabled(self, state: str) -> bool:
        """Return whether a lifecycle state is eligible under this explicit policy."""
        return state in ACCEPTED_USG_STATES or (
            self.include_candidate_states and state in CANDIDATE_USG_STATES
        )

    def source_access(
        self,
        record: Mapping[str, Any],
        sources_by_id: Mapping[str, Mapping[str, Any]],
        evidence_by_id: Mapping[str, Mapping[str, Any]],
    ) -> tuple[bool, str, tuple[str, ...]]:
        """Require all direct and evidence-derived source rights to permit local use."""
        source_ids = set(record.get("source_ids", ()))
        for evidence_id in record.get("evidence_ids", ()):
            evidence = evidence_by_id.get(evidence_id)
            if evidence is not None:
                source_id = evidence.get("source_id")
                if isinstance(source_id, str) and source_id:
                    source_ids.add(source_id)
        if not source_ids:
            return False, "missing_source_provenance", ()
        ordered = tuple(sorted(source_ids))
        if any(item not in sources_by_id for item in ordered):
            return False, "unknown_source_reference", ordered
        blocked = [
            item for item in ordered
            if sources_by_id[item].get("rights_status") not in self.allowed_local_rights
        ]
        if blocked:
            return False, "source_rights_not_permitted_for_local_import", ordered
        return True, "", ordered


__all__ = [
    "ACCEPTED_USG_STATES",
    "CANDIDATE_USG_STATES",
    "LOCAL_IMPORT_RIGHTS",
    "USGImportPolicy",
]
