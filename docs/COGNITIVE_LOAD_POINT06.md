# ORE Point 06 — Minimize Cognitive Load

This package implements the 16-step algorithm in the Library source file
`Point_06_Minimize_Cognitive_Load_Brain_Algorithm.md` as an inspectable Python
pipeline with separate functional and biological-reference outputs.

## What it does

The functional pipeline accepts a learning goal, ordered text units or source
text, explicit prerequisite edges, caller-provided prior knowledge, optional
examples, and optional behavioral observations. It then:

1. selects goal-relevant material;
2. estimates element interactivity and familiarity-adjusted demand;
3. retrieves only caller-supplied prerequisite memories;
4. segments material into source-ordered units;
5. forms reversible chunks from explicit/local coherence evidence;
6. orders acyclic prerequisite edges and reports cycles;
7. selects a principal subgoal;
8. estimates avoidable presentation costs;
9. builds an external outline/relation graph without inventing edges;
10. selects a bounded active working set and reports deferred prerequisites;
11. monitors performance and effort indicators;
12. adapts support and pace to the supplied capacity estimate;
13. builds guided prompts using only supplied examples;
14. integrates chunks using supplied relations;
15. tests retrieval before fading support;
16. verifies local, relational, transfer, and explanation evidence.

The package does not fabricate missing knowledge, infer unprovided semantic
relations, or treat self-reported effort as a direct measurement of neural load.

## API example

```python
from ore_cognitive_load import (
    Point06CognitiveLoadProcessor,
    Point06Config,
    Point06Request,
)

request = Point06Request(
    sequence_id="respiration-001",
    goal="explain how cellular respiration produces ATP",
    units=[
        "Glycolysis breaks down glucose.",
        "Electron carriers transport high-energy electrons.",
        "The electron transport chain builds a proton gradient.",
        "ATP synthase uses the gradient to produce ATP.",
    ],
    dependencies=[
        {"source": "u0001", "target": "u0002", "relation": "precedes"},
        {"source": "u0002", "target": "u0003", "relation": "supports"},
        {"source": "u0003", "target": "u0004", "relation": "enables"},
    ],
    prior_knowledge=["ATP is a cellular energy carrier."],
    metrics={
        "local_accuracy": 0.8,
        "relation_accuracy": 0.7,
        "transfer_accuracy": 0.6,
        "explanation_accuracy": 0.8,
    },
)

processor = Point06CognitiveLoadProcessor(
    Point06Config(simulate_biology=True)
)
result = processor.process(request)

assert len(result.operation_trace) == 16
print(result.comprehension)
print(result.biological["operation_count"])
```

Set `simulate_biology=False` to run the functional algorithm without the
optional distributed rate-coded reference network.

## SQLite annotations and run logging

The integration functions reuse the repository's existing `ore_logger`
package and the shared `ore_logger.sqlite3` database. They do not create a
second annotation catalogue:

```python
from ore_cognitive_load import register_annotations, record_result_to_sqlite

register_annotations(repository_root=".")
report = record_result_to_sqlite(result, repository_root=".")
print(report["database_summary"])
```

`register_annotations` indexes Point 06 modules, classes, functions, and
methods in the existing `annotations` table. `record_result_to_sqlite` writes
one run and 16 ordered operation events into the existing `runs` and `events`
tables. The implementation uses the repository logger's public
`LoggerDatabase` and `sync_repository` interfaces.

## Install and test

The root project's package discovery must include `ore_cognitive_load*`.
The local focused suite is:

```bash
python -m pytest tests/test_cognitive_load_point06.py -q
```

To run the example:

```bash
python -m ore_cognitive_load
```

## Module layout

- `ore_cognitive_load/contracts.py` — validated request, state, config, unit, and result contracts.
- `ore_cognitive_load/functional.py` — transparent mathematical and information-organization functions.
- `ore_cognitive_load/biology.py` — optional distributed rate-coded biological-reference network.
- `ore_cognitive_load/registry.py` — source-ordered 16-operation registry and equations.
- `ore_cognitive_load/pipeline.py` — end-to-end orchestration and shared SQLite logging.
- `ore_cognitive_load/annotations.py` — package annotation registration entry point.
- `tests/test_cognitive_load_point06.py` — focused functional, biological-reference, and logging tests.

## Scientific scope

This is a computational synthesis of behavioral and cognitive concepts. Its
scores, thresholds, graph operations, and population activations are explicit
engineering abstractions. The biological-reference populations are not direct
neural measurements, fitted physiological constants, or a validated model of
the complete human brain. Software tests establish implementation behavior,
not biological validity or learning effectiveness in human participants.