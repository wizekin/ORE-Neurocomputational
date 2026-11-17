# Point 8 — Cross-Stage Integration: A Recurrent Algorithm for Whole-Sentence and Discourse Comprehension

## Purpose and boundary

Point 8 is a coordination layer over the seven preceding processing parts. It is not an eighth compulsory stage that starts after memory and decision use. It coordinates partial outputs and feedback among sensory analysis, lexical access, syntax, semantic composition, discourse/pragmatics, propositional working memory, and memory/decision use while input is still arriving.

The implementation accepts partial stage outputs. It does not replace the specialized processors from Points 1–7 or pretend that all computations happen serially. A caller can pass results from whichever modules have already produced evidence. The recurrent state persists between calls to one integrator instance; call `reset()` to start a separate stream.

## 18 source-level steps

1. **Receive sensory evidence.** Add a time-indexed acoustic or visual increment, retain candidate alternatives, and preserve signal/source labels. The feature-extraction and Bayesian expressions in the source describe possible computational models, not literal neural arithmetic.
2. **Activate lexical candidates.** Score supplied word candidates with separate sensory, lexical, and contextual components. Context can bias a candidate but cannot overwrite the sensory record.
3. **Update grammatical hypotheses.** Retain one or more partial parses and their dependencies. Syntax evidence remains separately inspectable instead of being folded invisibly into semantic plausibility.
4. **Retrieve context-sensitive meanings.** Retain available word senses and their syntax/context support. Missing senses are not invented.
5. **Compose partial propositions.** Package supplied predicate/role structures and preserve open roles, alternatives, polarity, tense, and other operators where the upstream stage provides them.
6. **Update discourse and situation.** Merge entities, events, relations, temporal and spatial information, and current propositions. Conflicting context values are recorded rather than silently overwritten.
7. **Generate graded expectations.** Carry predictions supplied by the caller or an upstream predictor. The integration layer does not fabricate a language model or claim that one deterministic continuation is guaranteed.
8. **Detect typed prediction errors.** Compare paired observed and predicted values independently for sensory, lexical, syntactic, semantic, discourse, and pragmatic channels.
9. **Allocate attention/control.** Rank unresolved issues by uncertainty × impact × task relevance and distribute a bounded resource budget across them.
10. **Trigger targeted reanalysis.** Reopen only the stage named by a supplied revision candidate. A change is accepted only when its evidence score exceeds the prior score by the configured revision margin; displaced content is kept in the revision log.
11. **Perform pragmatic/social inference conditionally.** Evaluate caller-supplied inference candidates only when task demand or a pragmatic mismatch makes them relevant. Inferred content remains separate from literal propositions.
12. **Maintain working memory.** Refresh activation using persistence, new input, and interference:
   [
   A_i(t+1)=\rho A_i(t)+Input_i(t)-\kappa Interference_i(t)
   ]
   Keep the highest-priority items within a configurable engineering capacity and retain IDs for deprioritized items.
13. **Test comprehension adequacy.** Combine the available channel scores, subtract mismatch penalty, and compare the result with the task threshold. A critical unresolved dependency prevents finalization.
14. **Finalize a revisable proposition.** Return the best-supported literal proposition, inferred additions, remaining alternatives, and unresolved fields with provenance intact.
15. **Route downstream use.** Prepare optional discourse-continuation, question-answering, decision-support, episodic-memory, and semantic-memory paths. Routing does not itself execute an answer, action, or durable write.
16. **Reopen earlier stages after later evidence.** Apply explicit later revision candidates, preserve the prior snapshot and reason, then re-evaluate the interpretation.
17. **Prepare memory consolidation.** Package context, component provenance, and confidence for Point 7. Unverified semantic knowledge remains a candidate; Point 8 does not write directly to a durable memory database.
18. **Monitor outcomes and adapt strategy.** Record observed-versus-predicted outcome error and boundedly adjust attention and revision parameters. One correction does not automatically become a global linguistic rule.

## Mathematical model

The implementation exposes the following interpretable approximations:

[
\hat H_{1:t}=\arg\max_H P(H_{1:t}\mid X_{1:t},C_t,K,S)
]

This is a compact statement of inference under uncertainty; the code does not calculate a full global posterior over every possible interpretation.

[
J=L_{interpretation}+\lambda_1C_{memory}+\lambda_2C_{control}+\lambda_3C_{delay}
]

This models the trade-off among accuracy, memory load, control cost, and delay.

Candidate activation is represented as:

[
A_i(t)=\alpha A_i(t-1)+\beta SensoryFit_i+\gamma ContextFit_i+
\delta StructuralFit_i-\mu Conflict_i
]

Typed prediction error is kept as a vector rather than collapsed into a single undifferentiated score:

[
\boldsymbol\epsilon_t=(\epsilon_{sensory},\epsilon_{lexical},\epsilon_{syntactic},
\epsilon_{semantic},\epsilon_{discourse},\epsilon_{pragmatic})
]

Attention allocation uses:

[
Priority_i=Uncertainty_i\times Impact_i\times Relevance_i
\qquad \sum_i r_i\leq B
]

Reanalysis accepts a new candidate only when the evidence-weighted score compensates for revision cost:

[
Score(H_i)=\log P(X\mid H_i)+\log P(H_i\mid C)-\lambda Cost_{revision}(H_i)
]

Comprehension continues when:

[
Q_t\geq\theta_{task}
]

These are engineering approximations and testable hypotheses, not experimentally established equations for individual neurons.

## Recurrent feedback loops

The result exposes eight named interactions: context-to-perception/lexicon; syntax↔lexicon; syntax↔semantics; proposition↔context; prediction→error→reanalysis; working-memory↔control; pragmatic inference↔literal review; and memory→current comprehension.

The loop report says whether the available state engaged each interaction. The functional trace and state expose the actual candidates and changes; the loop labels are not claims that every brain system interacts on every sentence.

## Biological reference

The optional deterministic, current-based LIF reference contains ten functional populations: auditory/visual input, lexical, frontal-temporal language, semantic/conceptual, discourse/situation, prediction error, executive control, working memory, social inference, and medial-temporal memory. The output is explicitly labelled as simulated reference activity, not measured human data or a fitted anatomical circuit.

The biological descriptions are network-level and task-dependent. No one-to-one anatomical mapping is asserted. Prediction error, oscillatory coordination, synaptic plasticity, gain modulation, population coding, and recurrent feedback are presented as candidate principles rather than a unique mechanism for each Python operation.

## API

```python
from ore_language_comprehension import (
    Point8CrossStageIntegrator,
    Point8Request,
)

request = Point8Request(
    sequence_id="dialogue-001",
    increment={"text": "The child opened the door."},
    stage_outputs={
        "sensory": [{"candidate_id": "s1", "content": "The child opened the door", "confidence": 0.95}],
        "lexical": [{"candidate_id": "w1", "content": "opened", "confidence": 0.95}],
        "syntactic": [{"candidate_id": "g1", "content": "subject-verb-object", "confidence": 0.9}],
        "semantic": [{
            "candidate_id": "m1",
            "content": {"predicate": "OPEN", "semantic_roles": {"agent": "child", "theme": "door"}},
            "confidence": 0.95,
        }],
        "propositions": [{
            "candidate_id": "p1",
            "content": {"predicate": "OPEN", "semantic_roles": {"agent": "child", "theme": "door"}},
            "confidence": 0.95,
            "source": "explicit_linguistic_input",
        }],
    },
    context={"topic": "opening the door"},
    task={"requires_answer": True},
)

processor = Point8CrossStageIntegrator()
result = processor.process(request)
print(len(result.functional_trace))  # 18
print(result.finalized_interpretation)
print(result.feedback_loops)

# Later increments reuse the processor's current state.
next_result = processor.process(Point8Request(
    sequence_id="dialogue-002",
    increment={"text": "It was unlocked."},
))
```

To adapt actual result objects from preceding modules, use `Point8Request.from_stage_results(sequence_id, stage_results, ...)`. Supply stage outputs using explicit candidate fields when the upstream object uses a domain-specific schema. The integrator preserves the provided mappings and does not infer missing content from an opaque result.

Run:

```bash
python -m pytest tests/test_language_comprehension_point08.py -q
```

## Safeguards and limits

- Do not allow context to invent sensory input.
- Do not allow semantic plausibility to erase syntax or grammatical constraints.
- Bound candidate growth while retaining meaningful alternatives and revision history.
- Keep different error channels distinct.
- Trigger social inference conditionally, not for every sentence.
- Keep literal and inferred proposition content separate.
- Preserve evidence provenance and avoid confidence inflation from repeated activation.
- Do not convert unresolved inferences into verified memory facts.
- Keep prediction, model scores, and simulated LIF activity distinct from empirical neural measurements.
