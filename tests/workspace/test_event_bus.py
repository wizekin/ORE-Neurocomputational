"""Tests for selective notifications and subscriber failure isolation."""
from ore_workspace import FindingPublisher, SharedWorkspace


def test_only_subscribed_modules_are_notified_and_errors_are_isolated():
    """A raising subscriber cannot prevent another eligible consumer from receiving an event."""
    workspace = SharedWorkspace("boy opened door", "understand_sentence")
    received = []
    workspace.subscriptions.register("lexical_observer", ["lexical_hypotheses"])
    workspace.subscriptions.register("semantic_observer", ["semantic_hypotheses"])
    workspace.event_bus.subscribe("lexical_observer", lambda event: received.append(event.event_id))

    def fail(_event):
        """Represent an independently failing downstream consumer."""
        raise RuntimeError("consumer unavailable")

    workspace.event_bus.subscribe("semantic_observer", fail)
    result = workspace.publish(FindingPublisher(workspace, "lexical_processor").create_finding(
        "lexical_hypotheses", {"candidate": "PERSON"}, key="boy",
    ))
    assert result.changed_fields == ("lexical_hypotheses",)
    assert len(received) == 1
    assert [item.module_id for item in workspace.delivery_results] == ["lexical_observer"]
    assert workspace.delivery_results[0].delivered
    workspace.close()


def test_duplicate_event_delivery_is_idempotent_per_module():
    """Replaying one event ID does not invoke an already-attempted subscriber twice."""
    from ore_workspace import WorkspaceEvent

    workspace = SharedWorkspace("boy", "understand_sentence")
    workspace.subscriptions.register("lexical_observer", ["lexical_hypotheses"])
    received = []
    workspace.event_bus.subscribe("lexical_observer", lambda event: received.append(event.event_id))
    event = WorkspaceEvent(
        workspace_id=workspace.workspace_id, workspace_revision=0,
        event_type="workspace_field_updated", changed_fields=("lexical_hypotheses",),
        producer_module="lexical_processor", finding_ids=("finding_1",),
    )
    first = workspace.event_bus.publish(event)
    second = workspace.event_bus.publish(event)
    assert len(first) == 1 and first[0].delivered
    assert second == ()
    assert received == [event.event_id]
    workspace.close()