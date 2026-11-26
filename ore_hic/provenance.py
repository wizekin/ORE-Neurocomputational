"""Source lineage, rule traceability, artifact hashes, and audit history (spec 2.2, 7).

The ledger records every transformation between canonical artifacts. It never
mutates recorded entries: corrections are appended as new audit entries so the
full lineage from original idea to code and execution report stays
reconstructable (spec acceptance: "Full lineage ... is reconstructable").
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from .contracts import (
    ArtifactKind,
    ArtifactPointer,
    USGReference,
)

LEDGER_SCHEMA_VERSION = "0.1.0"


@dataclass(frozen=True, slots=True)
class ProvenanceEntry:
    """One immutable lineage step: artifact-to-artifact with rule citations."""

    entry_id: str
    stage: str
    step_id: str
    description: str
    input_digests: tuple[str, ...]
    output_digests: tuple[str, ...]
    rule_references: tuple[str, ...]
    usg_references: tuple[USGReference, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "stage": self.stage,
            "step_id": self.step_id,
            "description": self.description,
            "input_digests": list(self.input_digests),
            "output_digests": list(self.output_digests),
            "rule_references": list(self.rule_references),
            "usg_references": [ref.to_dict() for ref in self.usg_references],
        }


@dataclass(frozen=True, slots=True)
class AuditEntry:
    """An append-only audit event such as a gate decision or approval action."""

    audit_id: str
    event: str
    subject_digest: str
    details: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "audit_id": self.audit_id,
            "event": self.event,
            "subject_digest": self.subject_digest,
            "details": list(self.details),
        }


class ProvenanceLedger:
    """Append-only lineage and audit store for one compilation session."""

    def __init__(self) -> None:
        self._entries: list[ProvenanceEntry] = []
        self._audits: list[AuditEntry] = []
        self._counter = 0
        self._audit_counter = 0
        self._artifact_by_digest: dict[str, ArtifactPointer] = {}
        self._digest_of: dict[tuple[ArtifactKind, str], str] = {}

    # -- recording -----------------------------------------------------------

    def record(
        self,
        *,
        stage: str,
        step_id: str,
        description: str,
        inputs: Sequence[Any] = (),
        outputs: Sequence[Any] = (),
        rule_references: Iterable[str] = (),
        usg_references: Iterable[USGReference] = (),
    ) -> ProvenanceEntry:
        """Append one transformation entry; inputs and outputs keep their digests.

        Any object exposing ``content_digest`` may participate in the lineage;
        canonical artifacts additionally expose ``kind`` and ``artifact_id``.
        """
        self._counter += 1
        entry = ProvenanceEntry(
            entry_id=f"prov-{self._counter:05d}",
            stage=stage,
            step_id=step_id,
            description=description,
            input_digests=tuple(a.content_digest for a in inputs),
            output_digests=tuple(a.content_digest for a in outputs),
            rule_references=tuple(sorted(set(rule_references))),
            usg_references=tuple(usg_references),
        )
        self._entries.append(entry)
        for artifact in outputs:
            kind = getattr(artifact, "kind", None)
            pointer = ArtifactPointer(
                artifact_kind=kind.value if hasattr(kind, "value") else str(kind),
                artifact_id=getattr(artifact, "artifact_id", "unnamed"),
                version=int(getattr(artifact, "version", 1)),
                content_digest=artifact.content_digest,
            )
            self._artifact_by_digest.setdefault(artifact.content_digest, pointer)
        return entry

    def audit(self, event: str, subject_digest: str, *details: str) -> AuditEntry:
        """Append one audit event; used for gates, approvals, and rejections."""
        self._audit_counter += 1
        entry = AuditEntry(
            audit_id=f"audit-{self._audit_counter:05d}",
            event=event,
            subject_digest=subject_digest,
            details=tuple(details),
        )
        self._audits.append(entry)
        return entry

    # -- queries -------------------------------------------------------------

    @property
    def entries(self) -> tuple[ProvenanceEntry, ...]:
        return tuple(self._entries)

    @property
    def audits(self) -> tuple[AuditEntry, ...]:
        return tuple(self._audits)

    def entries_for_stage(self, stage: str) -> tuple[ProvenanceEntry, ...]:
        return tuple(e for e in self._entries if e.stage == stage)

    def artifact_pointer(self, content_digest: str) -> ArtifactPointer | None:
        return self._artifact_by_digest.get(content_digest)

    def lineage_of(self, content_digest: str) -> tuple[ProvenanceEntry, ...]:
        """Return the ordered entries producing or consuming ``content_digest``."""
        related: list[ProvenanceEntry] = []
        for entry in self._entries:
            if content_digest in entry.input_digests or content_digest in entry.output_digests:
                related.append(entry)
        return tuple(related)

    def reconstruct_lineage(self, final_digest: str) -> list[ProvenanceEntry]:
        """Walk backwards from ``final_digest`` to original inputs, depth-first."""
        chain: list[ProvenanceEntry] = []
        seen: set[str] = set()

        def _walk(digest: str) -> None:
            for entry in reversed(self._entries):
                if digest in entry.output_digests and entry.entry_id not in seen:
                    seen.add(entry.entry_id)
                    chain.append(entry)
                    for parent in entry.input_digests:
                        _walk(parent)

        _walk(final_digest)
        chain.reverse()
        return chain

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": LEDGER_SCHEMA_VERSION,
            "entries": [e.to_dict() for e in self._entries],
            "audits": [a.to_dict() for a in self._audits],
        }
