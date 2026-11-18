# ORE — Working-memory overload and information loss (Point 19)

## Scope

\`ore_working_memory_overload\` implements the 19 source operations in
\`Point_19_Working_Memory_Overload_and_Information_Loss_During_Complex_Interpretation.md\`.
It models the temporary maintenance and retrieval of information while a complex message is
being interpreted. It deliberately does not impose a universal word-count or item-count
capacity: estimated demand depends on unresolved dependencies, structural complexity, retrieval
distance, cue overlap, uncertainty, concurrent task load, chunking, and contextual support.

The implementation is a computational synthesis. Its biological track uses distributed,
functionally labelled rate-coded populations to make candidate interactions inspectable. These
signals are not measured human neural activity, fitted physiological parameters, or claims that
individual brain regions implement one software step each.

## The nineteen source-ordered operations

| Step | Operation | Main computation |
|---:|---|---|
| 1 | Establish goal | Normalize caller-supplied goal priorities with softmax-style weighting |
| 2 | Segment input | Split supplied text into word and punctuation units, or preserve caller units |
| 3 | Encode features | Create representations with observed features, position, confidence, and provenance |
| 4 | Build partial structure | Add only explicitly supplied unresolved dependencies |
| 5 | Maintain dependencies | Update accessibility using context support, refresh, distance decay, and similarity competition |
| 6 | Estimate load | Weighted sum of unresolved, structural, distance, interference, task, and uncertainty costs, less chunk/context relief |
| 7 | Chunk information | Group only explicitly tagged units; preserve reversible member links |
| 8 | Allocate attention | Temperature-scaled softmax over relevance, accessibility, and interference |
| 9 | Estimate interference | Similarity-weighted competition between simultaneously active representations |
| 10 | Retrieve by cue | Rank possible antecedents using required-feature match, cue match, accessibility, and task relevance |
| 11 | Integrate retrieval | Bind a candidate only when absolute evidence and retrieval confidence pass the threshold |
| 12 | Detect overload | Logistic estimate from load, slowing, reanalysis, error, and contextual support |
| 13 | Identify at-risk information | Rank representations by low accessibility, competition, open dependency demand, and uncertainty |
| 14 | Select recovery | Maximize expected accuracy benefit minus added load and time costs; apply only to existing evidence |
| 15 | Diagnose failure | Distinguish weak encoding, weak maintenance, retrieval interference, binding, and integration failures |
| 16 | Reconstruct supported content | Separate source-supported relations, caller-supplied hypotheses, and unknowns |
| 17 | Check coherence | Combine supplied grammar, semantic, and discourse checks and report omitted checks |
| 18 | Commit partial interpretation | Retain supported/inferred/unknown fields separately with provenance and confidence |
| 19 | Adapt difficulty | Bounded signed weight update from supplied prediction error and named difficulty drivers |

The equations in the operation registry are transparent engineering abstractions. In particular,
softmax, cosine/similarity, logistic overload estimates, and adaptive weight updates are not
claims that neurons literally calculate those expressions.

## Functional and biological tracks

The functional state is held in \`Point19State\`. \`Representation\` records include features,
accessibility, confidence, source, task relevance, position, chunk membership, and estimated
interference. \`Dependency\` records retain required features, retrieval cues, candidates, open/closed
state, confidence, and a failure label.

The optional \`DistributedWorkingMemoryNetwork\` contains coarse populations for:

- phonological and lexical maintenance;
- syntactic dependency workspace;
- semantic/event integration;
- parietal attention selection;
- prefrontal goal control;
- cingulo-opercular conflict/performance monitoring; and
- medial-temporal cue-based pattern completion.

Each operation advances the distributed rate-coded reference once. Sparse recurrent input,
competitive inhibition, logistic activation, and leaky persistence make the network dynamics
inspectable. These population labels describe candidate functional correspondences, not isolated
anatomical modules.

## Quick start

\`\`\`python
from ore_working_memory_overload import (
    Point19Config,
    Point19Request,
    Point19WorkingMemoryOverloadProcessor,
)

request = Point19Request(
    sequence_id="sentence-001",
    units=["Maya", "arrived", "."],
    unit_annotations=[
        {"features": {"syntactic_role": "subject", "referent": "Maya"}},
        {"features": {"syntactic_role": "predicate", "event": "arrive"}},
        {"features": {"punctuation": "period"}},
    ],
    dependencies=[{
        "identifier": "agent-of-arrival",
        "required_features": {"syntactic_role": "subject"},
        "cue_features": {"referent": "Maya"},
    }],
    goals={"identify_agent": 1.0, "understand_event": 0.8},
    context={"context_support": 0.5},
)

processor = Point19WorkingMemoryOverloadProcessor(
    Point19Config(simulate_biology=True)
)
result = processor.process(request)
print(len(result.operation_trace))            # 19
print(result.interpretation["supported"])
print(result.interpretation["unresolved"])
print(result.biological["operation_count"])   # 19
\`\`\`

For a text-only starting point, supply \`Point19Request(sequence_id="x", text="...")\`.
For successive increments that share a working context, reuse the same processor instance.
Set \`simulate_biology=False\` to run only the functional track.

## SQLite annotations and run logging

The package uses the repository's existing \`ore_logger\` implementation. Calling
\`register_annotations()\` scans the Point 19 package with its AST catalogue and stores module,
class, function, and method explanations in the shared \`annotations\` table. By default the
path is \`ore_logs/ore_logger.sqlite3\`; no separate annotation database is created.

\`record_result_to_sqlite(result)\` additionally writes all 19 operation outputs into the logger's
run/event tables. It registers annotations before writing the run. A custom database path can be
passed for tests or isolated experiments.

## Validation

Run the focused suite:

\`\`\`bash
python -m pytest tests/test_working_memory_overload_point19.py -q
\`\`\`

Scientific and implementation constraints:

- No missing word, referent, or relation is invented to fill a gap.
- An un-retrieved detail remains unknown; retrieval failure is not treated as proof the detail was absent.
- Relative candidate probability is not treated as absolute evidence strength.
- Recovery is constrained to stored input, supplied evidence, and explicit clarification options.
- Inferred reconstructions remain separate from directly supported relations.
- The processor does not perform durable episodic-memory writes.
