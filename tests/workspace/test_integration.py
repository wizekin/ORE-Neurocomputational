"""End-to-end tests of independently publishing ORE modules sharing one cognitive state."""
from ore_workspace import (
    FindingPublisher, GoalEvaluator, GoalOutcome, Provenance, SharedWorkspace,
)


def test_lexical_syntactic_semantic_and_working_memory_modules_coordinate():
    """Three analysis modules share findings through revisions while working memory stays selective."""
    workspace = SharedWorkspace("The boy opened the door.", "understand_sentence")
    lexical = FindingPublisher(workspace, "lexical_processor")
    syntax = FindingPublisher(workspace, "syntax_processor")
    semantic = FindingPublisher(workspace, "semantic_processor")
    working_memory = FindingPublisher(workspace, "working_memory")

    lexical_findings = [
        lexical.create_finding(
            "lexical_hypotheses", {"candidate": label, "word": word}, key=word,
            provenance=Provenance("current_input", span), confidence=confidence,
        )
        for word, label, span, confidence in (
            ("boy", "PERSON", (4, 7), 0.96),
            ("opened", "ACTION", (8, 14), 0.98),
            ("door", "OBJECT", (19, 23), 0.97),
        )
    ]
    lexical_results = [workspace.publish(item) for item in lexical_findings]
    assert all(item.decision.value == "accepted" for item in lexical_results)

    syntax_finding = syntax.create_finding(
        "syntactic_hypotheses", {"parse": "The boy=subject; opened=predicate; the door=object"},
        key="main_clause", dependencies=(lexical_findings[0].finding_id, lexical_findings[1].finding_id),
        provenance=Provenance("lexical_and_input_evidence"),
    )
    assert workspace.publish(syntax_finding).decision.value == "accepted"

    semantic_finding = semantic.create_finding(
        "semantic_hypotheses", {"candidate": "PERSON performs OPENING upon OBJECT"},
        key="sentence_event",
        dependencies=(lexical_findings[0].finding_id, lexical_findings[1].finding_id,
                      lexical_findings[2].finding_id, syntax_finding.finding_id),
        provenance=Provenance("lexical_and_syntactic_inference"),
        confidence=0.91,
    )
    assert workspace.publish(semantic_finding).decision.value == "accepted"

    memory_finding = working_memory.create_finding(
        "working_memory", {
            "item_id": "opening_event", "is_active": True,
            "retention_factors": {"goal": 1.0, "recency": 1.0, "context": 0.9, "unresolved": 0.8},
        }, key="opening_event", dependencies=(semantic_finding.finding_id,),
    )
    assert workspace.publish(memory_finding).decision.value == "accepted"

    snapshot = workspace.snapshot()
    assert len(snapshot.state["lexical_hypotheses"]) == 3
    assert len(snapshot.state["syntactic_hypotheses"]) == 1
    assert len(snapshot.state["semantic_hypotheses"]) == 1
    assert snapshot.state["working_memory"]["opening_event"]["is_active"]
    assert set(semantic_finding.dependencies).issubset(snapshot.state["accepted_updates"])
    assert workspace.revisions.latest().revision == workspace.revision

    remove = working_memory.create_finding(
        "working_memory", {"item_id": "opening_event", "action": "remove"}, key="opening_event",
    )
    workspace.publish(remove)
    after_removal = workspace.snapshot()
    assert not after_removal.state["working_memory"]["opening_event"]["is_active"]
    assert after_removal.state["semantic_hypotheses"]
    assert after_removal.state["lexical_hypotheses"]["boy"]

    evaluation = GoalEvaluator().evaluate(after_removal, {
        "lexical": 0.9, "syntactic": 0.9, "semantic": 0.95, "discourse": 0.5,
    })
    assert evaluation.outcome is GoalOutcome.COMPLETED
    workspace.close()


def test_missing_dependencies_are_deferred_then_reconsidered_after_dependency_arrives():
    """A semantic finding waits until its referenced syntax finding has been accepted."""
    workspace = SharedWorkspace("boy opened door", "understand_sentence")
    lexical = FindingPublisher(workspace, "lexical_processor").create_finding(
        "lexical_hypotheses", {"candidate": "PERSON"}, key="boy",
    )
    syntax = FindingPublisher(workspace, "syntax_processor").create_finding(
        "syntactic_hypotheses", {"parse": "subject predicate object"}, key="clause",
    )
    semantic = FindingPublisher(workspace, "semantic_processor").create_finding(
        "semantic_hypotheses", {"candidate": "person opens door"}, key="meaning",
        dependencies=(lexical.finding_id, syntax.finding_id),
    )
    deferred = workspace.publish(semantic)
    assert deferred.decision.value == "deferred"
    workspace.publish(lexical)
    # The first prerequisite alone is insufficient; the semantic finding remains pending.
    assert semantic.finding_id in workspace.pending_findings
    workspace.publish(syntax)
    # The second prerequisite triggers automatic reconsideration.
    assert semantic.finding_id not in workspace.pending_findings
    assert semantic.finding_id in workspace.snapshot().state["accepted_updates"]
    assert workspace.snapshot().state["semantic_hypotheses"]
    workspace.close()