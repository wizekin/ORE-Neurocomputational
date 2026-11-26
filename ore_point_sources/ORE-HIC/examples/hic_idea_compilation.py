"""Run a small, inspectable Point 16 idea-to-algorithm compilation example."""
from __future__ import annotations

from ore_hic import CompilationSession, HICCompiler, SessionSettings

IDEA = (
    "Assign each task to exactly one module. "
    "Modules have capacity limits. "
    "Tasks can only go to eligible modules. "
    "Prefer modules with higher capability and lower load. "
    "The total demand must not exceed module capacity. "
    "If no assignment is possible, report failure."
)


def main() -> None:
    """Compile the idea through all 52 steps with the spec's approval gates."""
    session = CompilationSession(user="demo-user", settings=SessionSettings(deterministic=True))
    compiler = HICCompiler(session)

    result = compiler.compile(IDEA)
    print(f"pass 1: {result.status}")

    if result.status == "AWAITING_CLARIFICATION":
        for question in result.clarification_questions:
            print(f"  Q{question.rank} (IG={question.information_gain_bits:.3f} bits): {question.question}")
        answers = {q.question_id: q.options[0] for q in result.clarification_questions}
        answers["as-model-linearity"] = "confirmed: costs and demands are additive"
        compiler.answer_clarifications(answers)
        result = compiler.compile(IDEA)
        print(f"pass 2: {result.status}")

    if result.status == "AWAITING_APPROVAL":
        compiler.approve(result.candidate_set, approved_by="demo-reviewer")
        result = compiler.compile(IDEA)
        print(f"pass 3: {result.status}")

    print(f"steps executed: {len(result.step_log)}")
    print(f"AlgoIR nodes: {len(result.algo_ir)}")
    print(f"selected candidate: {result.candidate_set.selected_id}")
    print(f"verification verdict: {result.verification_report.verdict}")
    print(f"failed properties: {[p.property_id for p in result.verification_report.failed_properties]}")
    print(f"sandbox outcomes: {[(o.fixture_name, o.status) for o in result.execution_report.outcomes]}")
    print(f"lineage entries: {len(result.lineage())}")
    print("\n--- generated Python (first 30 lines) ---")
    print("\n".join(result.code_artifact.source.splitlines()[:30]))


if __name__ == "__main__":
    main()
