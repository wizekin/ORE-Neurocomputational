# Language Comprehension Point 9 — Neural Architecture and Functional Division of Labour

Point 9 models the distributed neural architecture supporting language comprehension. It is not another serial linguistic stage. It coordinates interacting functional systems, each with a distinct computational contribution, while allowing task demands and evidence to change their relative engagement.

## Functional systems

The coordinator represents seven interacting system-level channels:

1. **Perceptual** — sensory evidence and signal quality.
2. **Language** — lexical, syntactic, phonological/orthographic and compositional linguistic representations.
3. **Conceptual knowledge** — semantic and world-knowledge candidates supplied by upstream components or the caller.
4. **Working memory** — active propositions, unresolved dependencies and currently accessible items.
5. **Mentalizing** — candidate models of speaker beliefs, intentions and perspectives.
6. **Multiple-demand / executive control** — conflict monitoring, effort, task control and resource allocation.
7. **Episodic/contextual memory** — context and prior-event information available to the current interpretation.

These are functional channels, not seven isolated brain organs. Real language comprehension is implemented by overlapping, recurrent and individually variable networks. A system name must not be interpreted as a one-region/one-function anatomical claim.

## Twenty coordinated operations

`Point9NeuralArchitectureCoordinator.process()` executes twenty ordered operations, recorded individually in `functional_trace`:

1. Receive upstream sensory and linguistic evidence with provenance.
2. Normalize supplied evidence into the seven functional channels.
3. Construct a sparse, directed functional communication graph.
4. Estimate system drives from supplied evidence and task demands.
5. Apply recurrent messages from connected systems.
6. Integrate conceptual knowledge candidates without fabricating missing knowledge.
7. Preserve literal propositions separately from inferred mental-state hypotheses.
8. Estimate perspective-taking / mentalizing demand.
9. Estimate executive demand from uncertainty, conflict, load and task pressure.
10. Update system activations using gated recurrent coupling.
11. Route typed prediction errors to the subsystem whose representation was challenged.
12. Maintain working-memory accessibility and unresolved dependencies.
13. Integrate supplied contextual/episodic candidates.
14. Compute functional-connectivity estimates only when caller-supplied time series are available.
15. Assess interpretation sufficiency while preserving critical unresolved items.
16. Prepare downstream answer, action and memory handoffs without performing durable writes.
17. Produce system-level and candidate-level diagnostic measures.
18. Produce optional leaky-integrate-and-fire (LIF) reference activity.
19. Record individual profile/localizer inputs and their provenance.
20. Return integrated functional state, trace, interpretation, routing and limitations.

The steps are a computational organization of the source's functional account, not a claim that the human brain executes twenty discrete serial neural events.

## Mathematical approximations

- **Evidence drive:** bounded weighted evidence aggregation, for example \(d_i = \sum_j w_{ij} e_j\).
- **Recurrent integration:** \(a_i(t+1) = (1-\\lambda)a_i(t) + \\lambda\\,\\sigma(b_i + d_i + g_i\\sum_j w_{ji}a_j(t))\), where \(g_i\) is a demand-dependent gate and weights represent functional coupling.
- **Sigmoid gating:** \(\\sigma(x)=1/(1+e^{-x})\), used for smooth recruitment rather than an all-or-none anatomical switch.
- **Mentalizing demand:** a bounded combination of explicit task requirements, perspective dependence, intention relevance and supplied social-context evidence.
- **Executive demand:** a bounded combination of uncertainty, competing interpretations, unresolved critical items, working-memory load and time pressure.
- **Prediction error:** typed mismatch between supplied prediction and observation; the type determines the functional subsystem receiving a revision notice.
- **Functional connectivity:** Pearson correlation \(r_{ij}=\\mathrm{corr}(x_i,x_j)\), calculated only from supplied time-series arrays. Correlation is not directionality or causal influence.
- **Individual localization:** where a caller provides a functional localizer, estimate a contrast such as \(\\bar{x}_{language}-\\bar{x}_{control}\), with standardized selectivity when enough samples are available. Without measured localizer data the result is explicitly labelled an approximate prior, not an individual measurement.
- **Interpretation sufficiency:** confidence and evidence support are evaluated alongside critical unresolved dependencies; high average confidence cannot erase a critical open issue.

These equations are engineering abstractions. They are not asserted to be the literal equations used by biological neurons or direct measurements of a participant's brain.

## Functional connectivity versus anatomical localization

The communication graph describes hypothesized functional information exchange. Its edge weights can be overridden by the caller or supplied individual profile. Edges are not white-matter tracts, and graph weights are not anatomical connectivity measurements.

Individual functional-localizer data are accepted as explicit input, including language/control measurements and a source label. The processor does not infer a person's activation pattern from age, sex, handedness, language background or other demographic fields. When no localizer is supplied, the result reports `approximate_prior_only`.

The optional time-series interface estimates pairwise correlations from caller-supplied data. It does not synthesize empirical fMRI, EEG or MEG observations.

## Optional LIF reference

Set `simulate_biology=True` to run the optional LIF population reference. It maintains separate simulated populations for the seven functional channels and records activity over the twenty operations. This is a transparent toy reference for population dynamics, not a fitted biophysical model, anatomical simulation or measured human dataset. The functional coordinator remains usable without it.

## API example

```python
from ore_language_comprehension import (
    Point9NeuralArchitectureCoordinator,
    Point9Request,
    register_point09_annotations,
)

request = Point9Request(
    sequence_id="sentence-001",
    stage_outputs={
        "perceptual": [{"candidate_id": "p1", "content": "auditory sentence", "confidence": 0.95}],
        "lexical": [{"candidate_id": "l1", "content": "opened", "confidence": 0.9}],
        "propositions": [{
            "candidate_id": "q1",
            "content": {"predicate": "OPEN", "semantic_roles": {"agent": "child", "theme": "door"}},
            "confidence": 0.95,
            "source": "explicit_linguistic_input",
        }],
    },
    context={"scene": "a child opens a door"},
    task={"requires_answer": True},
)

coordinator = Point9NeuralArchitectureCoordinator()
result = coordinator.process(request)
print(result.state["systems"])
print(result.functional_trace)
print(result.interpretation)
print(result.routing)

report = register_point09_annotations()
print(report["database_path"])
```

`Point9Request.from_stage_results(...)` adapts named outputs from Points 1–8. The adapter preserves supplied data and does not hallucinate missing representations. The shared annotation function uses the existing `ore_logger.annotations` SQLite catalogue; it does not create a separate Point 9 database.

## Safeguards and limits

- Literal linguistic content and mental-state hypotheses remain distinct.
- Mentalizing and executive support are conditionally recruited; a literal sentence does not automatically imply social inference or maximal control.
- Unprovided conceptual/world knowledge is never fabricated.
- Functional edges are not presented as anatomical tracts.
- Connectivity is not calculated without adequate caller-supplied time series.
- Demographic fields are not used to invent individual neural localization.
- Missing individual localizer data are reported as missing, not disguised as personalization.
- Durable memory is not written; memory and downstream operations are prepared as handoffs.
- LIF activity and model activation are simulations, not human neural measurements.

## Tests

```bash
python -m pytest tests/test_language_comprehension_point09.py -q
```
