"""Tests for active-memory retention scoring and constraint-aware goal completion."""
from ore_workspace import FindingPublisher, GoalEvaluator, GoalOutcome, SharedWorkspace, WorkingMemoryPolicy


def test_retention_scoring_and_capacity_select_high_priority_items():
    """Normalized weighted retention values determine active selection without deleting evidence."""
    policy = WorkingMemoryPolicy(capacity=1)
    items = {
        "boy": {"is_active": True, "retention_factors": {"goal": 1, "recency": 0.8}},
        "door": {"is_active": True, "retention_factors": {"goal": 0.2, "context": 0.1}},
    }
    assert policy.retention_score({"goal": 1, "recency": 1, "context": 1, "unresolved": 1}) == 1
    assert policy.select(items) == ("boy",)
    assert set(items) == {"boy", "door"}


def test_goal_score_cannot_hide_missing_critical_representations():
    """A high weighted score still requires lexical, syntactic and semantic representations."""
    workspace = SharedWorkspace("The boy opened the door.", "understand_sentence")
    evaluator = GoalEvaluator(threshold=0.5)
    evaluation = evaluator.evaluate(workspace.snapshot(), {
        "lexical": 1.0, "syntactic": 1.0, "semantic": 1.0, "discourse": 1.0,
    })
    assert evaluation.score == 1.0
    assert evaluation.outcome is GoalOutcome.NEEDS_MORE_PROCESSING
    assert "semantic_hypotheses" in evaluation.missing_requirements
    workspace.close()


def test_goal_can_complete_with_explicit_uncertainty():
    """The goal is complete with uncertainty when required structures exist but questions remain."""
    workspace = SharedWorkspace("The boy opened the door.", "understand_sentence")
    FindingPublisher(workspace, "lexical_processor").publish(
        "lexical_hypotheses", {"candidate": "PERSON"}, key="boy",
    )
    FindingPublisher(workspace, "syntax_processor").publish(
        "syntactic_hypotheses", {"parse": "subject-predicate-object"}, key="clause",
    )
    FindingPublisher(workspace, "semantic_processor").publish(
        "semantic_hypotheses", {"meaning": "PERSON opens OBJECT"}, key="event",
    )
    FindingPublisher(workspace, "cognitive_control").publish(
        "unresolved_questions", {"question_id": "q1", "resolved": False}, key="q1",
    )
    evaluator = GoalEvaluator(threshold=0.5)
    evaluation = evaluator.evaluate(workspace.snapshot(), {
        "lexical": 1.0, "syntactic": 1.0, "semantic": 1.0, "discourse": 0.0,
    })
    assert evaluation.outcome is GoalOutcome.COMPLETED_WITH_UNCERTAINTY
    assert evaluation.uncertainty_reasons == ("q1",)
    workspace.close()