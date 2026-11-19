"""Tests for weighted, subscription-driven and bounded module activation."""
from ore_workspace import FindingPublisher, ModuleScheduler, SharedWorkspace


def test_scheduler_activates_only_relevant_modules_and_publishes_findings():
    """A subscribed module processes a matching field event and returns a validated finding."""
    workspace = SharedWorkspace("The boy opened the door.", "understand_sentence")
    publisher = FindingPublisher(workspace, "semantic_processor")
    calls = []

    def semantic_handler(snapshot):
        """Produce a semantic interpretation from the current lexical evidence snapshot."""
        calls.append(snapshot.revision)
        return publisher.create_finding(
            "semantic_hypotheses", {"candidate": "PERSON performs ACTION on OBJECT"},
            dependencies=(snapshot.state["accepted_updates"][0],),
        )

    with ModuleScheduler(max_workers=2, max_rounds=2, max_activations=2) as scheduler:
        scheduler.register_module(
            workspace, "semantic_processor", semantic_handler,
            ["lexical_hypotheses"], sensitivity={"lexical_hypotheses": 2.0}, threshold=1.0,
        )
        lexical = FindingPublisher(workspace, "lexical_processor").create_finding(
            "lexical_hypotheses", {"candidate": "PERSON"}, key="boy",
        )
        workspace.publish(lexical)
        report = scheduler.run_until_idle(workspace)
    assert calls
    assert report.rounds >= 1
    assert report.publication_results
    assert workspace.snapshot().state["semantic_hypotheses"]
    workspace.close()


def test_scheduler_relevance_matrix_threshold_skips_unrelated_fields():
    """A field outside a module's subscription does not activate its callback."""
    workspace = SharedWorkspace("boy", "understand_sentence")
    calls = []
    with ModuleScheduler() as scheduler:
        scheduler.register_module(workspace, "syntax_processor", lambda snap: calls.append(snap.revision),
                                  ["syntactic_hypotheses"], threshold=0.1)
        activations, _ = scheduler.run_round(workspace, ["lexical_hypotheses"])
    assert not calls
    assert not activations
    workspace.close()


def test_scheduler_supports_async_module_handlers():
    """Coroutine-based modules execute in bounded worker threads and publish valid findings."""
    workspace = SharedWorkspace("boy", "understand_sentence")
    publisher = FindingPublisher(workspace, "lexical_processor")

    async def async_lexical(snapshot):
        """Return a lexical candidate after an asynchronous module operation."""
        return publisher.create_finding(
            "lexical_hypotheses", {"candidate": "PERSON"}, key="boy",
        )

    with ModuleScheduler(max_workers=1) as scheduler:
        scheduler.register_module(workspace, "lexical_processor", async_lexical, ["current_input"])
        from ore_workspace import WorkspaceEvent
        trigger = WorkspaceEvent(
            workspace_id=workspace.workspace_id, workspace_revision=0,
            event_type="workspace_field_updated", changed_fields=("current_input",),
            producer_module="external", finding_ids=(),
        )
        workspace.event_bus.publish(trigger)
        report = scheduler.run_until_idle(workspace)
    assert report.activations[0].queued
    assert workspace.snapshot().state["lexical_hypotheses"]["boy"]
    workspace.close()