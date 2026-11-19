"""Tests for workspace state isolation, field permissions and revision history."""
from concurrent.futures import ThreadPoolExecutor

from ore_workspace import (
    FindingPublisher, Provenance, SharedWorkspace, UpdateDecision,
)


def test_snapshot_is_detached_and_historical_revisions_are_reconstructable():
    """Readers cannot mutate live state and the original revision remains queryable."""
    workspace = SharedWorkspace("The boy opened the door.", "understand_sentence")
    publisher = FindingPublisher(workspace, "lexical_processor")
    finding = publisher.create_finding(
        "lexical_hypotheses", {"candidate": "PERSON"}, key="boy",
        provenance=Provenance("current_input", (4, 7)), confidence=0.96,
    )
    result = workspace.publish(finding)
    assert result.decision is UpdateDecision.ACCEPTED
    snapshot = workspace.snapshot()
    snapshot.state["lexical_hypotheses"]["boy"].clear()
    assert workspace.snapshot().state["lexical_hypotheses"]["boy"]
    assert workspace.snapshot(0).state["lexical_hypotheses"] == {}
    assert workspace.revisions.latest().revision == 1
    workspace.close()


def test_field_ownership_rejects_cross_module_writes():
    """Syntax modules cannot overwrite lexical evidence owned by lexical processing."""
    workspace = SharedWorkspace("The boy opened the door.", "understand_sentence")
    publisher = FindingPublisher(workspace, "syntax_processor")
    finding = publisher.create_finding("lexical_hypotheses", {"candidate": "PERSON"}, key="boy")
    result = workspace.publish(finding)
    assert result.decision is UpdateDecision.REJECTED
    assert finding.finding_id in workspace.rejected_findings
    assert not workspace.snapshot().state["lexical_hypotheses"]
    workspace.close()


def test_competing_hypotheses_are_preserved_and_linked_to_a_conflict():
    """Different candidates for the same lexical target remain visible in shared state."""
    workspace = SharedWorkspace("The boy opened the door.", "understand_sentence")
    publisher = FindingPublisher(workspace, "lexical_processor")
    first = publisher.create_finding("lexical_hypotheses", {"candidate": "PERSON"}, key="boy")
    second = publisher.create_finding("lexical_hypotheses", {"candidate": "ANIMAL"}, key="boy")
    assert workspace.publish(first).decision is UpdateDecision.ACCEPTED
    result = workspace.publish(second)
    assert result.decision is UpdateDecision.CONFLICT
    candidates = workspace.snapshot().state["lexical_hypotheses"]["boy"]
    assert {item["candidate"] for item in candidates} == {"PERSON", "ANIMAL"}
    conflict = workspace.conflicts.get(result.conflict_id)
    assert set(conflict.conflicting_findings) == {first.finding_id, second.finding_id}
    resolved = workspace.state_manager.resolve_conflict(
        conflict.conflict_id, preferred_finding_id=first.finding_id,
    )
    assert resolved.status == "resolved"
    assert resolved.resolved_at
    assert len(workspace.snapshot().state["lexical_hypotheses"]["boy"]) == 2
    workspace.close()


def test_stale_append_only_finding_rebases_but_mutable_update_is_deferred():
    """Append-only evidence survives a stale read while stale working-memory edits are deferred."""
    workspace = SharedWorkspace("The boy opened the door.", "understand_sentence")
    stale = FindingPublisher(workspace, "lexical_processor").create_finding(
        "lexical_hypotheses", {"candidate": "PERSON"}, key="boy", base_workspace_revision=0,
    )
    current = FindingPublisher(workspace, "lexical_processor").create_finding(
        "lexical_hypotheses", {"candidate": "ACTION"}, key="opened",
    )
    workspace.publish(current)
    rebased = workspace.publish(stale)
    assert rebased.decision is UpdateDecision.ACCEPTED
    assert rebased.rebased_from_revision == 0

    stale_memory = FindingPublisher(workspace, "working_memory").create_finding(
        "working_memory", {"item_id": "boy", "is_active": True}, key="boy",
        base_workspace_revision=0,
    )
    deferred = workspace.publish(stale_memory)
    assert deferred.decision is UpdateDecision.DEFERRED
    assert stale_memory.finding_id in workspace.pending_findings
    workspace.close()


def test_duplicate_publications_are_idempotent():
    """Equivalent content submitted twice does not append a second authoritative hypothesis."""
    workspace = SharedWorkspace("boy", "understand_sentence")
    publisher = FindingPublisher(workspace, "lexical_processor")
    first = publisher.create_finding("lexical_hypotheses", {"candidate": "PERSON"}, key="boy")
    duplicate = publisher.create_finding("lexical_hypotheses", {"candidate": "PERSON"}, key="boy")
    assert workspace.publish(first).decision is UpdateDecision.ACCEPTED
    assert workspace.publish(duplicate).decision is UpdateDecision.DUPLICATE
    assert len(workspace.snapshot().state["lexical_hypotheses"]["boy"]) == 1
    workspace.close()


def test_concurrent_append_only_publications_do_not_lose_updates():
    """Two modules publishing from the same revision are serialized and both findings persist."""
    workspace = SharedWorkspace("boy opened door", "understand_sentence")
    publisher = FindingPublisher(workspace, "lexical_processor")
    findings = [
        publisher.create_finding("lexical_hypotheses", {"candidate": candidate}, key=word,
                                 base_workspace_revision=0)
        for word, candidate in (("boy", "PERSON"), ("opened", "ACTION"), ("door", "OBJECT"))
    ]
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = tuple(pool.map(workspace.publish, findings))
    assert all(item.decision is UpdateDecision.ACCEPTED for item in results)
    assert set(workspace.snapshot().state["lexical_hypotheses"]) == {"boy", "opened", "door"}
    assert workspace.revision == 3
    workspace.close()