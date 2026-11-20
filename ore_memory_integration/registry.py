"""Mathematical operation registry for cross-structure memory integration."""
from __future__ import annotations

from typing import Any


MEMORY_INTEGRATION_OPERATIONS: tuple[dict[str, Any], ...] = (
    {"operation_id": "MEM-001", "name": "Initialize shared cognitive workspace", "mathematical_form": "W_0=(input,goal,revision=0)", "purpose": "Create a versioned communication state without creating a duplicate long-term memory store."},
    {"operation_id": "MEM-002", "name": "Maintain the current representation", "mathematical_form": "WM_0=WorkingMemoryProcessor(input,goal)", "purpose": "Process the current input before recall so retrieval cues are based on available evidence."},
    {"operation_id": "MEM-003", "name": "Construct multi-cue query", "mathematical_form": "q=(lexical,event,context,goal,temporal)", "purpose": "Combine only supplied or already computed cues; never fill missing event roles."},
    {"operation_id": "MEM-004", "name": "Rank episodic candidates", "mathematical_form": "R_e=(w_lL+w_rR+w_cC+w_gG+w_tT+w_sS)/sum(w)", "purpose": "Weight lexical, relation, context, goal, temporal and trace-strength compatibility."},
    {"operation_id": "MEM-005", "name": "Rank semantic candidates", "mathematical_form": "R_s=(v_cK+v_rR+v_gG+v_qQ+v_xX)/sum(v)", "purpose": "Weight conceptual, relational, goal, source-quality and contextual compatibility."},
    {"operation_id": "MEM-006", "name": "Enforce source independence", "mathematical_form": "E_g=max_{i:group(i)=g}(relevance_i*confidence_i)", "purpose": "Do not count a schema or analogy derived from one episode as an independent confirmation of that episode."},
    {"operation_id": "MEM-007", "name": "Publish normalized memory evidence", "mathematical_form": "W_{t+1}=Publish(W_t,MemoryEvidence)", "purpose": "Use the existing FindingPublisher, field permissions, revisions and event bus."},
    {"operation_id": "MEM-008", "name": "Select active memory references", "mathematical_form": "P_i=(.35G_i+.25R_i+.25C_i+.15U_i)/sum(weights)", "purpose": "Reuse the existing WorkingMemoryPolicy and retain references instead of copying the database."},
    {"operation_id": "MEM-009", "name": "Route evidence by cognitive role", "mathematical_form": "H_m=Adapter_m(E_{relevant})", "purpose": "Build explicit handoffs for semantic processing, schemas, analogy, reconstruction, discourse and control."},
    {"operation_id": "MEM-010", "name": "Evaluate independent evidence", "mathematical_form": "S(h)=sum_g(w_g E_g(h))-lambda*C(h)-mu*U(h)", "purpose": "Preserve conflict, uncertainty and source provenance; score is not a calibrated probability."},
    {"operation_id": "MEM-011", "name": "Bound recurrent integration", "mathematical_form": "0 <= feedback_rounds <= 1", "purpose": "Allow one memory-informed working-memory pass and prevent an unbounded callback loop."},
    {"operation_id": "MEM-012", "name": "Gate memory encoding", "mathematical_form": "Encode only if requested and source/confidence/scope checks pass", "purpose": "Perform optional encoding after recall and interpretation, never during the recall phase."},
)


def operation_registry_as_dicts() -> list[dict[str, Any]]:
    """Return a detached list of operation definitions for documentation and execution tracing."""
    return [dict(item) for item in MEMORY_INTEGRATION_OPERATIONS]
