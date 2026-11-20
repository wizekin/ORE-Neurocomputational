"""Regression tests for bounded recurrent feedback, evidence integrity, and model safety."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ore_contracts import LearningRequest, model_fingerprint
from ore_corrective_learning import (
    ClarificationOption,
    ConflictDimension,
    CorrectionRequest,
    CorrectiveConfig,
    FeedbackEvidence,
    KnowledgeHypothesis,
    RecurrentFeedbackConfig,
    RecurrentFeedbackController,
    RecurrentFeedbackRequest,
    register_recurrent_annotations,
)
from ore_corrective_learning.feedback_scheduler import FeedbackActionScheduler
from ore_corrective_learning.module_adapters import RecurrentModuleAdapters
from ore_corrective_learning.recurrent_controller import RECURRENT_FEEDBACK_OPERATIONS
from ore_logger.database import LoggerDatabase
from ore_n9 import N9FeedbackAdapter, N9Patch, N9Request


def _request(
    *,
    options: tuple[ClarificationOption, ...] = (),
    verified: str | None = None,
    compatibility: tuple[ConflictDimension, ...] = (),
    force_review: bool = False,
) -> RecurrentFeedbackRequest:
    """Build a two-candidate request with no implicit observation or evidence."""
    correction = CorrectionRequest(
        request_id="correction-001",
        hypotheses=(
            KnowledgeHypothesis(
                hypothesis_id="old",
                statement="The word means the familiar sense",
                prior=0.5,
                predictions={"answer-a": 0.8, "answer-b": 0.2},
                components={"predicate": "sense-a"},
                source="inferred",
            ),
            KnowledgeHypothesis(
                hypothesis_id="revised",
                statement="The word means the contextually supported sense",
                prior=0.5,
                predictions={"answer-a": 0.2, "answer-b": 0.8},
                components={"predicate": "sense-b"},
                source="inferred",
            ),
        ),
        current_hypothesis_id="old",
        clarification_options=options,
        verified_hypothesis_id=verified,
        context="ambiguous sentence context",
    )
    return RecurrentFeedbackRequest(
        request_id="feedback-001",
        correction_request=correction,
        input_snapshot={"text": "The ambiguous sentence", "source_id": "sentence-001"},
        initial_interpretation={"interpretation_id": "old", "statement": "The familiar sense"},
        context={"domain": "language"},
        compatibility_dimensions=compatibility,
        force_review=force_review,
    )


def _clarification_options(count: int = 1) -> tuple[ClarificationOption, ...]:
    """Create supplied, discriminating clarification options for the competing hypotheses."""
    return tuple(
        ClarificationOption(
            action_id=f"ask-source-{index}",
            kind="clarify",
            description=f"Ask for independent context detail {index}",
            response_models={
                "old": {"supports-old": 0.9, "supports-revised": 0.1},
                "revised": {"supports-old": 0.1, "supports-revised": 0.9},
            },
            cost=0.1,
        )
        for index in range(count)
    )


def _supporting_evidence(identifier: str = "independent-source-1") -> FeedbackEvidence:
    """Provide explicit hypothesis likelihoods from one independent, direct source."""
    return FeedbackEvidence(
        evidence_id=identifier,
        description="The surrounding sentence selects the revised semantic sense",
        likelihoods={"old": 0.05, "revised": 0.95},
        source_name="independent contextual reviewer",
        source_type="expert",
        reliability=1.0,
        relevance=1.0,
        diagnosticity=1.0,
        directness="direct",
        independence_key="reviewer-group-1",
        context="the same sentence with a broader discourse window",
    )


def _controller(**kwargs) -> RecurrentFeedbackController:
    """Create a deterministic functional controller with logging disabled for isolated tests."""
    return RecurrentFeedbackController(
        corrective_config=CorrectiveConfig(simulate_biology=False),
        register_logger_annotations=False,
        auto_log=False,
        **kwargs,
    )


def test_no_conflict_stops_without_unnecessary_recurrent_actions() -> None:
    """A lack of measured incompatibility must not trigger semantic feedback."""
    result = _controller().process(_request())

    assert result.status == "NO_CORRECTION_NEEDED"
    assert result.episode.current_iteration == 0
    assert result.episode.initial_interpretation["hypothesis_id"] == "old"
    assert result.episode.action_history == []
    assert [item["operation_id"] for item in result.operation_trace] == [
        item["operation_id"] for item in RECURRENT_FEEDBACK_OPERATIONS
    ]
    assert len(result.operation_trace) == 14
    assert result.diagnostics["model_updated"] is False


def test_missing_compatibility_dimensions_are_not_counted_as_conflict() -> None:
    """Unavailable module measurements are omitted rather than scored as incompatible."""
    request = _request(compatibility=(
        ConflictDimension("semantic_selection", None, 1.0, source="unavailable_semantic_module"),
    ))
    result = _controller().process(request)

    assert result.status == "NO_CORRECTION_NEEDED"
    assert result.diagnostics["initial_conflict_score"] == 0.0
    assert result.episode.conflict_history[0]["dimensions"] == []


def test_semantic_conflict_routes_feedback_and_reaches_verified_correction() -> None:
    """Independent likelihood-bearing feedback revises the posterior and resolves a semantic conflict."""
    calls = {"semantic": 0, "feedback": 0}

    def semantic_conflict(payload):
        """Return a measured conflict initially and improved compatibility after new evidence."""
        calls["semantic"] += 1
        compatibility = 0.05 if calls["semantic"] == 1 else 0.95
        return {
            "dimensions": [
                ConflictDimension(
                    "semantic_selection",
                    compatibility,
                    affected_modules=("semantic_processor",),
                    contradicting_evidence=("initial interpretation does not fit the sentence",),
                    missing_information=("independent contextual check",) if compatibility < 0.5 else (),
                )
            ]
        }

    option = _clarification_options()[0]

    def feedback_seeking(payload):
        """Offer one information-gathering action, then return its explicit likelihood evidence."""
        if "action" not in payload:
            return {"clarification_options": [option]}
        calls["feedback"] += 1
        return {"feedback_evidence": [_supporting_evidence()]}

    controller = _controller(module_handlers={
        "semantic_conflict": semantic_conflict,
        "feedback_seeking": feedback_seeking,
    })
    result = controller.process(_request(options=(option,), verified="revised"))

    assert result.status == "INDEPENDENTLY_VERIFIED"
    assert result.latest_corrective_result.selected_hypothesis_id == "revised"
    assert result.latest_corrective_result.posterior_probabilities["revised"] > 0.8
    assert result.episode.initial_interpretation["hypothesis_id"] == "old"
    assert result.episode.current_interpretation["interpretation_id"] == "revised"
    assert result.episode.correction_record["previous_interpretation_id"] == "old"
    assert result.episode.correction_record["revised_interpretation_id"] == "revised"
    assert result.episode.correction_record["model_updated"] is False
    assert calls["feedback"] == 1
    assert calls["semantic"] >= 2
    assert result.diagnostics["model_updated"] is False


def test_unverified_stable_interpretation_remains_uncertain() -> None:
    """Posterior stability without likelihood-bearing evidence is not accepted as correction."""
    option = _clarification_options()[0]

    def semantic_conflict(payload):
        """Keep the conflict present after every review attempt."""
        return {"dimensions": [ConflictDimension("semantic_selection", 0.0, affected_modules=("semantic_processor",))]}

    def feedback_seeking(payload):
        """Provide an action option but no evidence result."""
        if "action" not in payload:
            return {"clarification_options": [option]}
        return {"status": "completed_no_new_evidence"}

    result = _controller(module_handlers={
        "semantic_conflict": semantic_conflict,
        "feedback_seeking": feedback_seeking,
    }).process(_request(options=(option,)))

    assert result.status == "UNCERTAIN"
    assert result.episode.correction_record["verification_status"] == "not_independently_verified"
    assert result.episode.current_interpretation["status"] == "provisional"
    assert result.episode.correction_record["model_updated"] is False


def test_feedback_actions_are_fingerprint_deduplicated() -> None:
    """A failed action with unchanged context and evidence is not dispatched repeatedly."""
    request = _request(options=_clarification_options(1)).correction_request
    scheduler = FeedbackActionScheduler(RecurrentFeedbackConfig())
    probabilities = {"old": 0.5, "revised": 0.5}
    selected = scheduler.select(
        request,
        probabilities,
        target="old",
        context={"sentence": "same"},
        evidence_ids=(),
    )
    assert selected is not None
    scheduler.record_attempt(selected, "completed_no_new_evidence")
    assert scheduler.select(
        request,
        probabilities,
        target="old",
        context={"sentence": "same"},
        evidence_ids=(),
    ) is None
    assert scheduler.diagnostics()["failed_attempts_are_remembered"] is True


def test_maximum_iterations_stop_a_repeated_no_evidence_cycle() -> None:
    """The hard feedback iteration limit is enforced even if options remain."""
    options = _clarification_options(4)

    def semantic_conflict(payload):
        """Keep a semantic discrepancy open until the controller stops."""
        return {"dimensions": [ConflictDimension("semantic_selection", 0.0)]}

    def feedback_seeking(payload):
        """Return options on planning and no evidence when an action is executed."""
        if "action" not in payload:
            return {"clarification_options": list(options)}
        return {"status": "completed_no_new_evidence"}

    config = RecurrentFeedbackConfig(max_semantic_feedback_rounds=2, no_progress_patience=8)
    result = _controller(config=config, module_handlers={
        "semantic_conflict": semantic_conflict,
        "feedback_seeking": feedback_seeking,
    }).process(_request(options=options))

    assert result.status == "MAXIMUM_ITERATIONS_REACHED"
    assert result.episode.current_iteration == 2
    assert result.diagnostics["feedback_rounds"] <= 2


def test_conflicting_evidence_keeps_both_hypotheses_in_episode_history() -> None:
    """A revised posterior does not delete the original or competing interpretation."""
    option = _clarification_options()[0]

    def semantic_conflict(payload):
        """Report a semantic conflict once and its evidence-resolved state later."""
        compatibility = 0.05 if "prior_conflict_score" not in payload else 0.95
        return {"dimensions": [ConflictDimension("semantic_selection", compatibility)]}

    def feedback_seeking(payload):
        """Provide an option and then return evidence favouring the alternative interpretation."""
        return {"clarification_options": [option]} if "action" not in payload else {
            "feedback_evidence": [_supporting_evidence("supporting-alternative")]
        }

    result = _controller(module_handlers={
        "semantic_conflict": semantic_conflict,
        "feedback_seeking": feedback_seeking,
    }).process(_request(options=(option,)))

    ids = {item["hypothesis_id"] for item in result.episode.alternative_hypotheses}
    assert ids == {"old", "revised"}
    assert result.episode.initial_interpretation["statement"] == "The familiar sense"
    assert result.episode.correction_record["model_updated"] is False


def test_missing_modules_are_reported_as_unavailable_not_fabricated() -> None:
    """An unconfigured module must not return invented evidence or a synthetic result."""
    result = RecurrentModuleAdapters().invoke(
        "confirmation_bias",
        {"claim": "test"},
        {},
    )
    assert result.status == "unavailable"
    assert result.feedback_evidence == ()


@dataclass
class _DummyModel:
    """Minimal fingerprintable model used to test non-promotion of child updates."""

    weights: dict[str, float]

    def to_dict(self) -> dict[str, float]:
        """Expose a deterministic model representation for fingerprinting."""
        return dict(self.weights)


def test_unauthorized_learning_transaction_does_not_propose_or_promote_child() -> None:
    """The separate governance path rejects absent authorization and leaves the parent untouched."""
    parent = _DummyModel({"sense-a": 0.5})
    before = model_fingerprint(parent)
    learning_request = LearningRequest(
        parent_model_fingerprint=before,
        training_sample_ids=("train-001",),
        split_authorization="train-only",
        verified_targets={},
        verification_provenance={},
        update_channels=("semantic_relation",),
        explicitly_authorized=False,
        policy_reference="policy-unit-test",
        regression_suite_id="regression-unit-test",
        unsupervised_regime="declared-no-training",
    )
    proposal_called = {"value": False}

    def proposer(detached_parent, request):
        """Flag any unsafe model proposal invocation."""
        proposal_called["value"] = True
        return _DummyModel({"sense-a": 0.7})

    outcome = _controller().execute_authorized_learning_transaction(
        parent,
        learning_request,
        proposer,
        (("no-regression", lambda original, child: True),),
    )

    assert outcome.result.decision == "REJECTED"
    assert outcome.child_model is None
    assert proposal_called["value"] is False
    assert model_fingerprint(parent) == before


def test_n9_reprocessing_forces_learning_authorization_off_on_detached_request() -> None:
    """N9 feedback re-entry uses a copied request and cannot mutate the caller's learning flag."""
    class FakeResult:
        """Provide the minimal N9 result mapping consumed by the feedback adapter."""

        def to_dict(self):
            """Expose a model-safe status snapshot."""
            return {
                "sequence_id": "n9-sequence",
                "learning": {"status": "skipped_without_explicit_authorization", "parent_model_mutated": False},
                "diagnostics": {"integrity_valid": True},
            }

    class FakeProcessor:
        """Capture the detached N9 request supplied to process()."""

        def __init__(self):
            self.request = None

        def process(self, request):
            """Capture the N9 feedback request and return a minimal result."""
            self.request = request
            return FakeResult()

    processor = FakeProcessor()
    request = N9Request(
        sequence_id="n9-sequence",
        patches=(N9Patch(patch_id="patch-1", features=np.asarray([0.1, 0.8], dtype=np.float32)),),
        learning_authorized=True,
    )
    original_context = dict(request.context)
    result = N9FeedbackAdapter(processor).reprocess(
        request,
        {"hypothesis_id": "revised", "statement": "revised interpretation"},
        ("semantic_relation",),
    )

    assert result["status"] == "completed"
    assert result["model_learning_authorized"] is False
    assert processor.request is not request
    assert processor.request.learning_authorized is False
    assert request.learning_authorized is True
    assert request.context == original_context


def test_logger_indexes_every_new_controller_symbol_and_links_class_methods(tmp_path) -> None:
    """Register recurrent controller classes and functions in the shared SQLite annotation table."""
    root = Path(__file__).resolve().parents[1]
    database_path = tmp_path / "ore_logger.sqlite3"
    report = register_recurrent_annotations(database_path=database_path, repository_root=root)
    database = LoggerDatabase(database_path)
    controller_class = database.get_annotation(
        "ore_corrective_learning.recurrent_controller",
        "RecurrentFeedbackController",
        "class",
        "ore_corrective_learning/recurrent_controller.py",
    )
    process_method = database.get_annotation(
        "ore_corrective_learning.recurrent_controller",
        "RecurrentFeedbackController.process",
        "function",
        "ore_corrective_learning/recurrent_controller.py",
    )
    adapter_class = database.get_annotation(
        "ore_n9.feedback_adapter",
        "N9FeedbackAdapter",
        "class",
        "ore_n9/feedback_adapter.py",
    )
    feedback_class = database.get_annotation(
        "ore_feedback_seeking.pipeline",
        "Point07FeedbackProcessor",
        "class",
        "ore_feedback_seeking/pipeline.py",
    )

    assert report["symbols_written"] > 0
    assert controller_class is not None
    assert process_method is not None
    assert adapter_class is not None
    assert feedback_class is not None


def test_inference_does_not_mutate_supplied_model_or_claim_learning() -> None:
    """A recurrent correction record is an inference result, never a model-weight update."""
    parent = _DummyModel({"semantic": 0.25})
    before = model_fingerprint(parent)
    result = _controller().process(_request(
        compatibility=(ConflictDimension("semantic_selection", 0.0),),
    ))

    assert model_fingerprint(parent) == before
    assert result.episode.correction_record["model_updated"] is False
    assert result.diagnostics["model_updated"] is False
