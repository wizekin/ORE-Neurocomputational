"""Validated strategy registry (registry layer; section 4 contract).

``register(artifact, approval) -> RegistryRecord`` requires an explicit approval;
missing manifests, hash mismatches and duplicate hashes are rejected or deduplicated
(acceptance tests 23, 24, 28). Production selection is restricted to APPROVED /
CANARY / ACTIVE strategies — GENERATED, SANDBOX_TESTED, VALIDATED and REJECTED
strategies can never be selected in production (acceptance test 26). Records are
immutable: state changes append new records and keep the full history.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ..domain.strategy import PRODUCTION_ELIGIBLE_STATES, StrategyCandidate
from ..verification.promotion_policy import (
    ApprovalRecord, PromotionPolicy, RejectionRecord,
)
from .artifact_store import ArtifactStore, StrategyArtifact, StrategyManifest
from .versioning import Versioning


@dataclass(frozen=True, slots=True)
class RegistryRecord:
    """One immutable registry record for one strategy version."""

    strategy_id: str
    content_hash: str
    state: str
    version: int
    manifest_digest: str
    registered_at_utc: str
    approval: ApprovalRecord | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "strategy_id": self.strategy_id, "content_hash": self.content_hash,
            "state": self.state, "version": self.version,
            "manifest_digest": self.manifest_digest,
            "registered_at_utc": self.registered_at_utc,
            "approval": self.approval.as_dict() if self.approval else None,
        }


class ValidatedStrategyRegistry:
    """Registry of validated, approved strategies (section 4 contract)."""

    def __init__(self, *, artifact_store: ArtifactStore | None = None) -> None:
        self.store = artifact_store or ArtifactStore()
        self.versioning = Versioning()
        self.promotion_policy = PromotionPolicy()
        self._records: dict[str, RegistryRecord] = {}
        self._history: list[RegistryRecord] = []

    # ---------------------------------------------------------------- registration
    def register(
        self,
        artifact: StrategyArtifact,
        approval: ApprovalRecord | None = None,
        *,
        candidate: StrategyCandidate | None = None,
    ) -> tuple[RegistryRecord, str]:
        """Register one approved artifact; returns ``(record, note)``.

        Rejects artifacts whose manifest is missing or whose hash does not verify;
        duplicate content hashes are deduplicated (the original record is returned).
        """
        if artifact is None or artifact.manifest is None:
            raise ValueError("registry rejects artifacts without a manifest")
        # Integrity re-verification (acceptance test 28).
        from ..domain.genome import content_digest
        if content_digest(dict(artifact.payload)) != artifact.content_hash:
            raise ValueError(
                "registry rejects artifacts whose content hash does not verify"
            )
        if artifact.content_hash in self._records:
            existing = self._records[artifact.content_hash]
            return existing, "duplicate: content hash already registered"
        if approval is None:
            raise ValueError(
                "registry rejects strategies without an explicit approval record"
            )
        strategy_id = candidate.candidate_id if candidate is not None else artifact.content_hash[:16]
        state = candidate.state if candidate is not None else "VALIDATED"
        version_info = self.versioning.register(
            strategy_id, artifact.content_hash,
            change_kind="initial",
            notes=f"approved by {approval.approver}",
        )
        record = RegistryRecord(
            strategy_id=strategy_id,
            content_hash=artifact.content_hash,
            state=state,
            version=version_info.version,
            manifest_digest=artifact.manifest.digest(),
            registered_at_utc=version_info.registered_at_utc,
            approval=approval,
        )
        self._records[artifact.content_hash] = record
        self._history.append(record)
        self.store.save(artifact)
        return record, "registered"

    # ------------------------------------------------------------------- promotion
    def promote(
        self,
        candidate: StrategyCandidate,
        target_state: str,
        evidence: Mapping[str, Any],
    ) -> tuple[StrategyCandidate, Any]:
        """Apply one lifecycle transition through the promotion policy."""
        new_candidate, decision = self.promotion_policy.apply(
            candidate, target_state, evidence,
        )
        if decision.granted and new_candidate.content_hash in self._records:
            record = self._records[new_candidate.content_hash]
            from dataclasses import replace
            updated = replace(record, state=new_candidate.state)
            self._records[new_candidate.content_hash] = updated
            self._history.append(updated)
        return new_candidate, decision

    def reject(
        self, candidate: StrategyCandidate, reason: str, *, stage: str = "",
    ) -> tuple[StrategyCandidate, Any]:
        """Reject a candidate with a mandatory reason (safety failures included)."""
        record = RejectionRecord(reason=reason, stage=stage)
        return self.promotion_policy.apply(candidate, "REJECTED", {"rejection": record})

    # -------------------------------------------------------------------- selection
    def records(self) -> list[RegistryRecord]:
        return list(self._records.values())

    def history(self) -> list[RegistryRecord]:
        return list(self._history)

    def production_eligible(self) -> list[RegistryRecord]:
        """Only APPROVED / CANARY / ACTIVE strategies are production-eligible."""
        return [
            record for record in self._records.values()
            if record.state in PRODUCTION_ELIGIBLE_STATES
        ]

    def select_for_context(self, context: Mapping[str, Any]) -> RegistryRecord:
        """Select one production strategy by task context (acceptance test 26).

        Raises when no APPROVED / CANARY / ACTIVE strategy exists — never falls back
        to an unapproved candidate.
        """
        eligible = self.production_eligible()
        if not eligible:
            raise LookupError(
                "no production-eligible strategy is registered; only APPROVED, "
                "CANARY or ACTIVE strategies can be selected in production"
            )
        preferred = context.get("preferred_strategy_id")
        if preferred:
            for record in eligible:
                if record.strategy_id == preferred:
                    return record
        return max(eligible, key=lambda record: record.version)

    def artifact_for(self, record: RegistryRecord) -> StrategyArtifact:
        """Load the verified artifact behind one registry record."""
        return self.store.load(record.content_hash)

    def manifest_for(self, record: RegistryRecord) -> StrategyManifest:
        return self.artifact_for(record).manifest


__all__ = ["RegistryRecord", "ValidatedStrategyRegistry"]
