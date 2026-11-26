"""ORE integration: workspace publication, write governance, SQLite audit (Steps 44-46).

The reasoning engine is read-only against authoritative USG by default. Derived
facts are proposed with derivation provenance and assumption metadata; promotion
requires the configured validation and approval policy. Verified findings,
conflicts, counterexamples, optimization proposals, and proof artifacts are
published to the shared cognitive workspace as separate typed entries, and every
execution is persisted to a replayable SQLite audit record.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .contracts import FormalResult, ResultKind
from .exceptions import GovernanceViolationError
from .version import FRCE_PRODUCER

# Workspace target fields per result category. The shared workspace has a fixed
# field vocabulary; FRCE entries never create new fields, they publish typed
# artifacts into existing fields with distinct finding types.
_FIELD_BY_KIND: Mapping[ResultKind, str] = {
    ResultKind.RETRIEVED_SOURCE_FACT: "semantic_hypotheses",
    ResultKind.RULE_DERIVED_INFERENCE: "semantic_hypotheses",
    ResultKind.SATISFIABILITY_RESULT: "artifact_references",
    ResultKind.OPTIMIZATION_RESULT: "artifact_references",
    ResultKind.MACHINE_CHECKED_PROOF: "artifact_references",
    ResultKind.EMPIRICAL_OBSERVATION: "semantic_hypotheses",
}

_FINDING_TYPE_BY_KIND: Mapping[ResultKind, str] = {
    ResultKind.RETRIEVED_SOURCE_FACT: "frce_source_fact",
    ResultKind.RULE_DERIVED_INFERENCE: "frce_inferred_fact",
    ResultKind.SATISFIABILITY_RESULT: "frce_satisfiability_result",
    ResultKind.OPTIMIZATION_RESULT: "frce_optimization_proposal",
    ResultKind.MACHINE_CHECKED_PROOF: "frce_proof_artifact",
    ResultKind.EMPIRICAL_OBSERVATION: "frce_empirical_observation",
}

_CONFLICT_FIELD = "unresolved_questions"
_CONFLICT_FINDING_TYPE = "frce_conflict"
_COUNTEREXAMPLE_FINDING_TYPE = "frce_counterexample"
_ASSUMPTION_FINDING_TYPE = "frce_assumption_set"


@dataclass(frozen=True, slots=True)
class PublicationRecord:
    """One workspace publication with its typed category (Step 44)."""

    finding_type: str
    workspace_field: str
    payload: Mapping[str, Any]
    accepted: bool
    detail: str


class WorkspacePublicationAdapter:
    """Publish typed FRCE results into the shared cognitive workspace (Step 44).

    Publication distinguishes inferred, satisfiable, optimized, and proved
    result types; an unverified hypothesis is never presented as a verified
    fact — the finding type and the payload status carry that distinction, and
    the workspace's own permission policy still gates every write.
    """

    def __init__(self, producer_module: str = FRCE_PRODUCER) -> None:
        self.producer_module = producer_module

    def _publish_finding(
        self,
        workspace: Any,
        permission_policy: Any,
        field_name: str,
        finding_type: str,
        payload: Mapping[str, Any],
        provenance_rules: Sequence[str],
        confidence: float | None,
    ) -> PublicationRecord:
        """Create and submit one finding through the workspace state manager."""
        from ore_workspace.models import Finding, FindingTarget, Provenance
        from ore_workspace.publication import FindingPublisher

        if permission_policy is not None and not permission_policy.can_write(
            self.producer_module, field_name
        ):
            return PublicationRecord(
                finding_type=finding_type,
                workspace_field=field_name,
                payload=dict(payload),
                accepted=False,
                detail=(
                    f"permission policy does not grant {self.producer_module} "
                    f"write access to {field_name}"
                ),
            )
        FindingPublisher(workspace, self.producer_module)  # validates producer binding
        finding = Finding(
            workspace_id=workspace.workspace_id,
            producer_module=self.producer_module,
            finding_type=finding_type,
            target=FindingTarget(field=field_name, key=payload.get("result_id")),
            payload=dict(payload),
            provenance=Provenance(
                source="formal_reasoning_engine",
                rules_applied=tuple(provenance_rules),
                notes="published by the FRCE result-governance path",
            ),
            base_workspace_revision=workspace.snapshot().revision,
            confidence=confidence,
        )
        validation = workspace.publish(finding)
        accepted = validation.decision.value in {"accepted", "merged"}
        return PublicationRecord(
            finding_type=finding_type,
            workspace_field=field_name,
            payload=dict(payload),
            accepted=accepted,
            detail=f"workspace decision: {validation.decision.value}",
        )

    def publish_result(
        self,
        workspace: Any,
        result: FormalResult,
        permission_policy: Any | None = None,
    ) -> PublicationRecord:
        """Publish one typed result (Step 44)."""
        field_name = _FIELD_BY_KIND[result.kind]
        finding_type = _FINDING_TYPE_BY_KIND[result.kind]
        payload = result.to_dict()
        payload["epistemic_category"] = result.kind.value
        payload["status"] = result.status.value
        provenance_rules = tuple(result.statement_ir_refs)
        return self._publish_finding(
            workspace,
            permission_policy,
            field_name,
            finding_type,
            payload,
            provenance_rules,
            confidence=None,
        )

    def publish_conflict(
        self,
        workspace: Any,
        conflict: Mapping[str, Any],
        permission_policy: Any | None = None,
    ) -> PublicationRecord:
        """Publish one preserved conflict as a separate typed entry (Step 44)."""
        return self._publish_finding(
            workspace,
            permission_policy,
            _CONFLICT_FIELD,
            _CONFLICT_FINDING_TYPE,
            dict(conflict),
            tuple(conflict.get("minimal_evidence", ()) or ()),
            None,
        )

    def publish_counterexample(
        self,
        workspace: Any,
        counterexample: Mapping[str, Any],
        statement_refs: Sequence[str] = (),
        permission_policy: Any | None = None,
    ) -> PublicationRecord:
        """Publish one counterexample as a separate typed entry (Step 44)."""
        return self._publish_finding(
            workspace,
            permission_policy,
            "artifact_references",
            _COUNTEREXAMPLE_FINDING_TYPE,
            dict(counterexample),
            tuple(statement_refs),
            None,
        )

    def publish_assumptions(
        self,
        workspace: Any,
        assumptions: Iterable[Mapping[str, Any]],
        permission_policy: Any | None = None,
    ) -> PublicationRecord:
        """Publish the assumption registry as a distinct typed entry (Step 44)."""
        payload = {"assumptions": [dict(a) for a in assumptions]}
        return self._publish_finding(
            workspace,
            permission_policy,
            "artifact_references",
            _ASSUMPTION_FINDING_TYPE,
            payload,
            (),
            None,
        )


# ---------------------------------------------------------------------------
# Knowledge-write governance (Step 45)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PromotionProposal:
    """A proposed derived fact awaiting governance approval (Step 45)."""

    proposal_id: str
    statement: str
    derivation_refs: tuple[str, ...]
    assumption_names: tuple[str, ...]
    confidence_note: str
    source: str = FRCE_PRODUCER

    def __post_init__(self) -> None:
        if not self.proposal_id.strip() or not self.statement.strip():
            raise ValueError("promotion proposals need non-empty id and statement")
        object.__setattr__(self, "derivation_refs", tuple(self.derivation_refs))
        object.__setattr__(self, "assumption_names", tuple(self.assumption_names))


@dataclass(frozen=True, slots=True)
class PromotionDecision:
    """The governance decision over one proposal."""

    proposal_id: str
    approved: bool
    reason: str
    approved_by: str | None = None


class PromotionPolicy:
    """Approve or deny promotion of derived facts into authoritative knowledge."""

    def __init__(
        self,
        *,
        require_explicit_approval: bool = True,
        approvers: Iterable[str] = (),
    ) -> None:
        self.require_explicit_approval = require_explicit_approval
        self._approvers = frozenset(approvers)

    def approve(
        self, proposal: PromotionProposal, approver: str | None = None,
    ) -> PromotionDecision:
        """Decide one proposal; the default policy denies silent promotion."""
        if not self.require_explicit_approval:
            return PromotionDecision(
                proposal_id=proposal.proposal_id,
                approved=True,
                reason="policy allows automatic promotion",
                approved_by=approver,
            )
        if approver is None or approver not in self._approvers:
            return PromotionDecision(
                proposal_id=proposal.proposal_id,
                approved=False,
                reason=(
                    "promotion requires an authorized approver; derived facts stay "
                    "proposals with derivation provenance"
                ),
                approved_by=approver,
            )
        return PromotionDecision(
            proposal_id=proposal.proposal_id,
            approved=True,
            reason="approved by an authorized approver",
            approved_by=approver,
        )


class KnowledgeWriteGovernance:
    """Enforce the read-only USG boundary and the promotion workflow (Step 45)."""

    def __init__(self, promotion_policy: PromotionPolicy | None = None) -> None:
        self.promotion_policy = promotion_policy or PromotionPolicy()

    def propose_derived_fact(
        self,
        statement: str,
        derivation_refs: Sequence[str],
        assumption_names: Sequence[str],
        *,
        proposal_id: str,
        confidence_note: str = "",
    ) -> PromotionProposal:
        """Create a promotion proposal; nothing is written to USG here."""
        return PromotionProposal(
            proposal_id=proposal_id,
            statement=statement,
            derivation_refs=tuple(derivation_refs),
            assumption_names=tuple(assumption_names),
            confidence_note=confidence_note,
        )

    def submit(self, proposal: PromotionProposal, approver: str | None = None) -> PromotionDecision:
        """Submit a proposal through the promotion policy."""
        return self.promotion_policy.approve(proposal, approver)

    @staticmethod
    def attempt_direct_usg_write(*_args: Any, **_kwargs: Any) -> None:
        """Always refuses: the engine has no write path into authoritative USG."""
        raise GovernanceViolationError(
            "the formal reasoning engine is read-only against authoritative USG "
            "storage; derived knowledge moves through the promotion workflow only"
        )


# ---------------------------------------------------------------------------
# SQLite audit log (Step 46)
# ---------------------------------------------------------------------------

_SECRET_MARKERS = ("secret", "token", "password", "api_key", "apikey", "credential")

_AUDIT_SCHEMA = """
CREATE TABLE IF NOT EXISTS frce_execution_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id TEXT NOT NULL,
    fingerprint TEXT NOT NULL,
    created_at TEXT NOT NULL,
    schema_version TEXT NOT NULL,
    ir_version TEXT NOT NULL,
    ir_hash TEXT NOT NULL,
    input_fact_ids TEXT NOT NULL,
    input_rule_ids TEXT NOT NULL,
    assumptions TEXT NOT NULL,
    solver_versions TEXT NOT NULL,
    solver_config TEXT NOT NULL,
    resource_use TEXT NOT NULL,
    result_kind TEXT NOT NULL,
    status TEXT NOT NULL,
    proof_refs TEXT NOT NULL,
    artifact_hashes TEXT NOT NULL,
    error TEXT,
    trace TEXT NOT NULL
);
"""


def _sanitize(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Drop secret-like keys before anything reaches the audit store."""
    clean: dict[str, Any] = {}
    for key, value in payload.items():
        lowered = str(key).lower()
        if any(marker in lowered for marker in _SECRET_MARKERS):
            clean[key] = "<redacted>"
        else:
            clean[key] = value
    return clean


class SQLiteAuditLog:
    """Replayable audit records for every FRCE execution (Step 46)."""

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(str(self.database_path), check_same_thread=False)
        self._connection.executescript(_AUDIT_SCHEMA)
        self._connection.commit()

    def record_run(self, record: Mapping[str, Any]) -> int:
        """Persist one execution record and return its row id."""
        clean = _sanitize(record)
        required = ("request_id", "fingerprint", "status")
        for name in required:
            if name not in clean:
                raise ValueError(f"audit record requires {name!r}")
        cursor = self._connection.execute(
            """
            INSERT INTO frce_execution_log (
                request_id, fingerprint, created_at, schema_version, ir_version,
                ir_hash, input_fact_ids, input_rule_ids, assumptions,
                solver_versions, solver_config, resource_use, result_kind, status,
                proof_refs, artifact_hashes, error, trace
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                clean["request_id"],
                clean["fingerprint"],
                clean.get("created_at", ""),
                clean.get("schema_version", ""),
                clean.get("ir_version", ""),
                clean.get("ir_hash", ""),
                json.dumps(clean.get("input_fact_ids", []), sort_keys=True),
                json.dumps(clean.get("input_rule_ids", []), sort_keys=True),
                json.dumps(clean.get("assumptions", []), sort_keys=True),
                json.dumps(clean.get("solver_versions", []), sort_keys=True),
                json.dumps(clean.get("solver_config", {}), sort_keys=True),
                json.dumps(clean.get("resource_use", {}), sort_keys=True),
                clean.get("result_kind", ""),
                clean["status"],
                json.dumps(clean.get("proof_refs", []), sort_keys=True),
                json.dumps(clean.get("artifact_hashes", []), sort_keys=True),
                clean.get("error"),
                json.dumps(clean.get("trace", []), sort_keys=True),
            ),
        )
        self._connection.commit()
        return int(cursor.lastrowid)

    def get_run(self, request_id: str) -> dict[str, Any] | None:
        """Return the most recent audit record for one request id."""
        row = self._connection.execute(
            "SELECT * FROM frce_execution_log WHERE request_id = ? ORDER BY id DESC LIMIT 1",
            (request_id,),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_dict(row)

    def list_runs(self, limit: int = 50) -> tuple[dict[str, Any], ...]:
        """Return recent audit records in insertion order."""
        rows = self._connection.execute(
            "SELECT * FROM frce_execution_log ORDER BY id LIMIT ?", (int(limit),)
        ).fetchall()
        return tuple(self._row_to_dict(row) for row in rows)

    @staticmethod
    def _row_to_dict(row: Sequence[Any]) -> dict[str, Any]:
        keys = (
            "id", "request_id", "fingerprint", "created_at", "schema_version",
            "ir_version", "ir_hash", "input_fact_ids", "input_rule_ids",
            "assumptions", "solver_versions", "solver_config", "resource_use",
            "result_kind", "status", "proof_refs", "artifact_hashes", "error",
            "trace",
        )
        record = dict(zip(keys, row))
        for name in (
            "input_fact_ids", "input_rule_ids", "assumptions", "solver_versions",
            "solver_config", "resource_use", "proof_refs", "artifact_hashes", "trace",
        ):
            try:
                record[name] = json.loads(record[name])
            except (TypeError, ValueError):
                record[name] = record[name]
        return record

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self._connection.close()


__all__ = [
    "PublicationRecord",
    "WorkspacePublicationAdapter",
    "PromotionProposal",
    "PromotionDecision",
    "PromotionPolicy",
    "KnowledgeWriteGovernance",
    "SQLiteAuditLog",
    "FRCE_PRODUCER",
]
