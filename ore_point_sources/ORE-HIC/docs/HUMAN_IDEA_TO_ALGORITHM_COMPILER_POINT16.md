# Human Idea-to-Algorithm Compiler (HIC) — Point 16: Semantic Formalization and AlgoIR

Implementation of the Point 16 integrated algorithm: a compiler-like subsystem that transforms an informal human idea into an explicit, inspectable, reusable computational specification, then into candidate algorithms and verified executable artifacts.

    Human input -> USG semantic interpretation -> ambiguity and context analysis ->
    requirements specification -> mathematical model -> canonical AlgoIR ->
    candidate algorithm discovery -> independent verification ->
    target-language generation -> sandboxed testing -> human approval and
    controlled execution -> evaluation and learning.

Scope statement: HIC is not natural-language-to-code generation. Its primary task is intent formalization: identify what the person means, expose ambiguity and missing requirements, distinguish facts from assumptions, construct mathematical models where appropriate, and preserve a traceable chain from the original input to every generated operation. No stage silently converts uncertainty into fact, and no generated code is treated as correct merely because it compiles or runs.

## Module architecture (spec Section 6 -> repository layout)

| Spec boundary | Implementation |
|---|---|
| `ore_hic.input` | `ore_hic/input.py` — text, speech-transcript, file, diagram adapters, optional constrained BCI adapter (disabled without measured signals and a decoder) |
| `ore_hic.session` | `ore_hic/session.py` — sessions, immutable `InputRecord`, artifact versioning |
| `ore_hic.semantic` | `ore_hic/semantic.py` — concept, relation, attribute extraction with USG rule traceability |
| `ore_hic.ambiguity` | `ore_hic/ambiguity.py` — entropy, expected information gain, ranked questions, explicit assumptions |
| `ore_hic.requirements` | `ore_hic/requirements.py` — objectives, interfaces, actors, predicates, constraints, state machine, failures, termination, acceptance criteria |
| `ore_hic.math` | `ore_hic/mathmodel.py` (named `mathmodel` to avoid stdlib-shadowing ambiguity in tooling) — typed variables, equations, objectives, assumption validation, tool selection, complexity |
| `ore_hic.algoir` | `ore_hic/algoir.py` — typed node schema, validation, serialization, human-readable rendering |
| `ore_hic.discovery` | `ore_hic/discovery.py` — candidate families, plans, template synthesis, comparison, selection |
| `ore_hic.verification` | `ore_hic/verification.py` — static AST checks, Z3 proof obligations (optional), property tests with shrinking, differential and metamorphic testing, traceability |
| `ore_hic.codegen` | `ore_hic/codegen.py` — deterministic Python generation with source maps |
| `ore_hic.sandbox` | `ore_hic/sandbox.py` — restricted subprocess execution with resource limits and scrubbed environment |
| `ore_hic.provenance` | `ore_hic/provenance.py` — append-only lineage and audit ledger |
| `ore_hic.learning` | `ore_hic/learning.py` — evaluation against acceptance criteria, metrics, safe proposals |
| ore_usg adapter | `ore_hic/usg_adapter.py` — starter registry plus a read-only bridge to `ore_usg.USGKnowledgeIndex` |
| ORE cognitive coordinator | `ore_hic/coordinator.py` — the 52-step `HICCompiler` with gates and approvals |
| CROS adapter | `ore_hic/cros_adapter.py` — bounded per-stage resource budgets |

Supporting files: `contracts.py` (shared enums, digests, canonical artifact base), `registry.py` (the 52-operation `OPERATION_REGISTRY`), `annotations.py` (ore_logger indexing), `segmentation.py` (Steps 5-7).

## The 52 steps and the gates

Phases and steps follow spec Section 5 exactly; `HICCompiler.compile()` logs every step (`HIC01`..`HIC52`) with status EXECUTED/BLOCKED and records it in the provenance ledger.

- Phase A (Steps 1-7): session, immutable input with integrity digest, modality and quality flags, loss-free normalization as a derived view, segmentation into claims/intentions with source spans, deliverable identification, project-context resolution (absent facts stay UNKNOWN).
- Phase B (Steps 8-16): USG-grounded concept extraction (unresolved words stay unresolved), predicate-frame relation extraction, attributes with units and modalities, class-hierarchy resolution, semantic graph, competing interpretations retained with evidence (`P(h|X,C) ∝ P(X|h,C)P(h|C)` as a ranking, not proof), ambiguity detection, question ranking by `IG(q) = H(H) − E_a[H(H|a)]`, explicit assumptions with source/confidence/impact.
- Phase C (Steps 17-26): objective (primary/secondary/prohibited), typed interfaces with units and domains, actor permissions, pre/postconditions (`{P} A {Q}`), invariants, hard constraints (never weighted) vs soft preferences, state machine with guarded transitions and recovery, failure cases with reversibility and compensation, termination criteria, measurable acceptance criteria.
- Phase D (Steps 27-33): structure identification; the spec 3.3 assignment ILP (`maximize Σ_ij x_ij(αc_ij − βl_j − γr_ij)` s.t. exactly-one, capacity, eligibility) with typed variables, derivation-tracked equations, validated assumptions, fitting tools, and complexity estimates with input assumptions and confidence.
- Phase E (Steps 34-39): AlgoIR instantiation with all 18 required node types and full metadata; well-formedness validation (undefined symbols, dependency cycles outside declared iterative constructs, branch completeness, loop termination, units, reachability, outputs, constraint scopes, unresolved assumptions); human-readable explanation rendered from the canonical IR.
- Phase F (Steps 40-44): greedy-by-task, greedy-by-pair, exhaustive-optimum, and Hungarian candidates; plans sharing one external specification; template synthesis; systematic comparison; incompatible candidates (Hungarian under general capacities) are REJECTED regardless of speed.
- Phase G (Steps 45-49): static analysis (approved imports only, no eval/exec/open/socket/subprocess), proof obligations (Z3 discharges the capacity step invariant when installed; unproven scope is recorded), seeded property tests with greedy counterexample shrinking, differential testing against the exhaustive reference, metamorphic relations (permutation invariance, conservation), bidirectional traceability with reported gaps.
- Phase H (Steps 50-52): deterministic Python generation with line-level source maps, sandboxed execution (CPU/memory/file-size/wall limits, scrubbed env, static policy pre-check), evaluation against acceptance criteria, versioned lineage, and learning proposals that never mutate approved artifacts.

Gates between phases (spec Section 7): blocking ambiguity or an unconfirmed high-impact assumption pauses the pipeline with `AWAITING_CLARIFICATION`; malformed AlgoIR is rejected before synthesis; candidates failing verification are rejected; code generation requires an approval bound to the candidate-set semantic digest — a changed artifact requires renewed approval.

## Canonical artifacts

`InputRecord`, `SemanticGraph`, `AmbiguitySet`, `RequirementSpec`, `MathModel`, `AlgoIR`, `CandidateSet`, `VerificationReport`, `CodeArtifact`, `ExecutionReport`, `ApprovalRecord`, `LearningRecord`. Each is versioned, deterministically serializable (canonical JSON), and digestable. `content_digest` covers the full payload; `semantic_digest` excludes housekeeping ids and volatile runtime measurements so that deterministic recompilations of the same versioned input are comparable and approvals stay bound to content.

## Usage

    from ore_hic import CompilationSession, HICCompiler, SessionSettings

    session = CompilationSession(user="analyst", settings=SessionSettings(deterministic=True))
    compiler = HICCompiler(session)
    result = compiler.compile(
        "Assign each task to exactly one module. Modules have capacity limits. "
        "Tasks can only go to eligible modules. Prefer modules with higher "
        "capability and lower load. If no assignment is possible, report failure."
    )
    if result.status == "AWAITING_CLARIFICATION":
        answers = {q.question_id: q.options[0] for q in result.clarification_questions}
        answers["as-model-linearity"] = "confirmed"
        compiler.answer_clarifications(answers)
        result = compiler.compile(idea)
    if result.status == "AWAITING_APPROVAL":
        compiler.approve(result.candidate_set, approved_by="reviewer")
        result = compiler.compile(idea)

    print(result.status)                  # COMPLETED
    print(result.explanation[:200])       # human-readable AlgoIR explanation
    print(result.code_artifact.source)    # generated, source-mapped Python
    print(len(result.lineage()))          # full input->code lineage

One-shot convenience: `from ore_hic import compile_idea; compile_idea(idea_text)`.

## Verification evidence produced per run

- static findings (approved-import policy, unsafe calls),
- proof obligations with discharge status, method, assumptions and evidence,
- property results with cases run and minimized counterexamples when they fail,
- differential results (greedy vs exhaustive reference: false negatives quantified, never hidden),
- metamorphic results (permutation invariance, conservation),
- traceability report (requirement -> IR node -> test, uncovered requirements, unproven properties, gaps).

Execution reports explicitly state that runtime success is not proof of semantic correctness.

## Dependencies

Runtime: Python 3.10+ standard library only (the package does not require numpy for the compiler itself). Optional: `z3-solver` discharges the arithmetic step-invariant proof obligation (otherwise recorded UNPROVEN); `hypothesis` can complement the built-in property runner. Install extras with `pip install -e ".[hic]"`.

## Integration with ORE-Neurocomputational

- Read-only USG grounding: `HICUSGAdapter(external_index=USGKnowledgeIndex(resource))` delegates lookups to `ore_usg`; matches against the external resource are labeled hypothesis mappings (`RELATED_TO`, rule `HIC-USG-R07`), never promoted to authoritative concepts.
- Repository conventions followed: flat `ore_hic` package, `OPERATION_REGISTRY` with `OperationSpec`, `register_annotations()` for ore_logger, pytest tests under `tests/`, ruff-clean at line length 100.
- The CROS adapter is the seam for the repository-wide resource-coordination subsystem; `pyproject.toml` includes `ore_hic*` in the package list.

## Limitations (kept explicit, per spec Section 1)

- The semantic interpreter is deterministic and rule-based with an explicit starter ontology; it is not a general-purpose NLP parser. Unknown text degrades to clarification questions rather than invented semantics.
- The worked mathematical family is the spec 3.3 task-to-module assignment ILP; other structures are detected and reported honestly as UNRECOGNIZED rather than forced into a solver.
- The sandbox enforces a static policy pre-check plus OS rlimits and a scrubbed environment; it is not a namespace/kernel sandbox. Network egress is blocked by policy pre-check, not by network namespaces.
- Code generation targets the approved AlgoIR subset for the assignment family; Rust/C++ generators are future work (spec Stage 6).
- Hungarian assignment is retained in the candidate set but rejected as incompatible with the general (multi-capacity) specification, with its reason recorded.
