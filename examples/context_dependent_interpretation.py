"""Runnable example for Point 12 context-dependent interpretation."""

from __future__ import annotations

from ore_context_dependent_interpretation import (
    CandidateHypothesis,
    ContextDependentInterpretationProcessor,
    ContextEvidence,
    ContextState,
    InputEvent,
)


def main() -> None:
    """Run an ambiguous-word example with a finance-oriented preceding discourse."""
    state = ContextState(
        discourse_history=[
            ContextEvidence(
                evidence_id="turn-1",
                source_type="discourse",
                content="We need to withdraw cash and check the account balance at the bank.",
                concepts=("bank", "cash", "account", "withdrawal"),
                goal_tags=("find_financial_service",),
                provenance="previous_speaker_turn",
            )
        ],
        current_goal="find_financial_service",
    )
    event = InputEvent(
        event_id="event-bank-closed",
        content="The bank is closed.",
        candidates=(
            CandidateHypothesis(
                candidate_id="financial",
                meaning="a financial institution is closed",
                cue_terms=("cash", "account", "money", "deposit"),
            ),
            CandidateHypothesis(
                candidate_id="river",
                meaning="the side of a river is closed",
                cue_terms=("river", "shore", "fishing", "water"),
            ),
        ),
    )
    result = ContextDependentInterpretationProcessor(state=state).process(event)
    print(result.output)
    print("Uncertainty (nats):", round(result.uncertainty, 3))
    print("Operation count:", len(result.operation_trace))


if __name__ == "__main__":
    main()
