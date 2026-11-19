"""Run a small, inspectable Point 01 active-engagement example."""
from __future__ import annotations

from ore_active_engagement import ActiveEngagementProcessor, ActiveEngagementRequest


def build_demo_request() -> ActiveEngagementRequest:
    """Build a cell-membrane example with an explicitly learner-generated explanation."""
    return ActiveEngagementRequest(
        learner_id="demo-learner",
        goal="understand how the cell membrane regulates selective transport",
        material=(
            "The cell membrane forms a boundary around the cell. "
            "It regulates which substances enter and leave. "
            "This selective transport helps maintain stable internal conditions."
        ),
        learner_explanation=(
            "The cell membrane controls what enters and leaves the cell, "
            "which helps keep the internal environment stable."
        ),
        required_concepts=("membrane", "substances", "enter", "leave"),
        listener_level="novice",
        enable_teaching=True,
        enable_imagery=True,
    )


def main() -> None:
    """Run the active-engagement demonstration and print its episode summary."""
    request = build_demo_request()
    result = ActiveEngagementProcessor(request.learner_id).process(request)
    print(result.output)
    print(f"Operations completed: {len(result.operation_trace)}")


if __name__ == "__main__":
    main()
