"""Shared fixtures for the ore_hic test suite."""
from __future__ import annotations

from ore_hic import CompilationSession, HICCompiler, SessionSettings

ASSIGNMENT_IDEA = (
    "Assign each task to exactly one module. "
    "Modules have capacity limits. "
    "Tasks can only go to eligible modules. "
    "Prefer modules with higher capability and lower load. "
    "The total demand must not exceed module capacity. "
    "If no assignment is possible, report failure."
)


def fast_settings(**overrides) -> SessionSettings:
    """Deterministic settings with a small property-test budget."""
    values = {
        "deterministic": True,
        "property_test_cases": 8,
        "property_test_seed": 7,
    }
    values.update(overrides)
    return SessionSettings(**values)


def build_compiler(**overrides) -> HICCompiler:
    session = CompilationSession(user="hic-test-user", settings=fast_settings(**overrides))
    return HICCompiler(session)


def run_to_completion(compiler: HICCompiler, idea: str = ASSIGNMENT_IDEA):
    """Drive the pipeline through the clarification and approval gates."""
    result = compiler.compile(idea)
    if result.status == "AWAITING_CLARIFICATION":
        answers = {q.question_id: q.options[0] for q in result.clarification_questions}
        answers["as-model-linearity"] = "confirmed: costs and demands are additive"
        compiler.answer_clarifications(answers)
        result = compiler.compile(idea)
    if result.status == "AWAITING_APPROVAL":
        compiler.approve(result.candidate_set, approved_by="test-reviewer")
        result = compiler.compile(idea)
    return result
