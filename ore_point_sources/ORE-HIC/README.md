# ORE HIC — Human Idea-to-Algorithm Compiler (Point 16)

Full implementation of the Point 16 integrated algorithm — *Human Idea-to-Algorithm
Compiler, Semantic Formalization and AlgoIR* — for the ORE-Neurocomputational
repository. This archive contains **only the algorithm's files**: the `ore_hic`
package, its test suite, and its documentation.

## What it does

HIC compiles an informal human idea into an explicit, inspectable, reusable
computational specification and then into verified executable artifacts:

    input -> USG semantic interpretation -> ambiguity analysis -> requirements ->
    mathematical model (spec 3.3 assignment ILP) -> canonical AlgoIR ->
    candidate discovery -> independent verification -> Python generation ->
    sandboxed testing -> human approval -> controlled execution -> learning.

All 52 pipeline steps (spec Section 5) are implemented across eight phases (A-H)
with the approval gates of spec Section 7. Uncertainty is never silently
converted into fact; generated code is never assumed correct merely because it
runs.

## Archive layout

    ore_hic/                                        the algorithm package (19 modules)
        contracts.py        shared enums, digests, canonical artifact base
        session.py          Steps 1-2: sessions, immutable InputRecord
        input.py            Steps 2-4: adapters, modality, loss-free normalization
        segmentation.py     Steps 5-7: claims/intentions, deliverable, context
        usg_adapter.py      USG concept/relation/rule resolution (+ ore_usg bridge)
        semantic.py         Steps 8-13: semantic graph with rule traceability
        ambiguity.py        Steps 14-16: entropy, information gain, assumptions
        requirements.py     Steps 17-26: RequirementSpec, state machine, failures
        mathmodel.py        Steps 27-33: variables, equations, ILP, complexity
        algoir.py           Steps 34-39: AlgoIR schema, validation, explanation
        discovery.py        Steps 40-44: candidates, synthesis, comparison
        verification.py     Steps 45-49: static checks, proofs, property tests
        codegen.py          Step 50: Python generation with source maps
        sandbox.py          Step 51: restricted execution, resource limits
        learning.py         Step 52: evaluation, versioned learning
        provenance.py       lineage, rule traceability, audit ledger
        cros_adapter.py     bounded resource budgets per stage
        coordinator.py      the 52-step HICCompiler with gates and approvals
        registry.py         OPERATION_REGISTRY of the 52 steps
        annotations.py      ore_logger annotation helper
    tests/                                          122 pytest tests (10 files)
    docs/HUMAN_IDEA_TO_ALGORITHM_COMPILER_POINT16.md  full algorithm documentation

## Installation into ORE-Neurocomputational

1. Unzip this archive into the repository root; it merges cleanly:

       unzip ore-hic-point16.zip -d /path/to/ORE-Neurocomputational

2. Add the package to `pyproject.toml` if it is not yet present:

       [tool.setuptools.packages.find]
       include = [..., "ore_hic*"]

   Optional verification extras:

       [project.optional-dependencies]
       hic = ["z3-solver>=4.12", "hypothesis>=6.100"]

3. Run the tests (Python 3.10+; no runtime dependency beyond the standard
   library — z3 and hypothesis are optional):

       python -m pytest tests/test_ore_hic_*.py -q

## Quick start

    from ore_hic import CompilationSession, HICCompiler, SessionSettings

    session = CompilationSession(user="analyst", settings=SessionSettings(deterministic=True))
    compiler = HICCompiler(session)
    idea = ("Assign each task to exactly one module. Modules have capacity limits. "
            "Tasks can only go to eligible modules. Prefer modules with higher "
            "capability and lower load. If no assignment is possible, report failure.")

    result = compiler.compile(idea)
    if result.status == "AWAITING_CLARIFICATION":
        answers = {q.question_id: q.options[0] for q in result.clarification_questions}
        answers["as-model-linearity"] = "confirmed"
        compiler.answer_clarifications(answers)
        result = compiler.compile(idea)
    if result.status == "AWAITING_APPROVAL":
        compiler.approve(result.candidate_set, approved_by="reviewer")
        result = compiler.compile(idea)

    print(result.status)                # COMPLETED
    print(result.explanation)           # human-readable AlgoIR explanation
    print(result.code_artifact.source)  # generated, source-mapped Python
    print([e.step_id for e in result.lineage()])  # full input->code lineage

## Key guarantees (tested against spec Section 9)

- Original input preserved byte-for-byte with integrity digests; every derived
  fragment keeps its source span.
- Semantic items cite the USG concept/rule supporting them or stay unresolved;
  competing interpretations are preserved, never collapsed.
- Clarification questions ranked by expected information gain and impact;
  blocking ambiguity and unconfirmed high-impact assumptions stop the pipeline
  before code generation.
- Hard constraints can never be overridden by soft objective weights.
- The state machine rejects illegal transitions; loops require termination
  arguments; dependency cycles outside declared iterative constructs are
  rejected; AlgoIR serialization round-trips.
- Candidates violating hard constraints are rejected even if faster; the
  Hungarian family is rejected as incompatible with the general specification.
- Property tests discover and minimize counterexamples (verified with a
  deliberately broken candidate); differential testing quantifies greedy
  false-negatives against the exhaustive reference.
- Generated code passes the approved-import static policy, carries a complete
  source map, and runs in a sandbox with CPU/memory/wall limits and a scrubbed
  environment; malicious code is blocked before execution.
- Approvals bind to the exact artifact semantic digest; changed artifacts
  require renewed approval; learning proposals never mutate approved artifacts.
- Recompiling the same versioned input under deterministic settings yields
  identical semantic digests and byte-identical generated source.

## Limitations (explicit, per spec)

Rule-based semantic interpretation with an explicit starter ontology (not a
general NLP parser); the worked mathematical family is the spec 3.3 assignment
ILP; the sandbox is a static policy pre-check plus OS rlimits, not a kernel
namespace sandbox; Python is the only generation target (Rust/C++ are spec
Stage 6 future work).

See `docs/HUMAN_IDEA_TO_ALGORITHM_COMPILER_POINT16.md` for the complete
architecture, step-by-step mapping, and usage reference.
