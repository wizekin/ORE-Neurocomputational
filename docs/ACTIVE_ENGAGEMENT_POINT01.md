# Point 01 — Active Engagement: Implementation Notes

## Scope

This package implements Point_01_Active_Engagement_Brain_Algorithm.md as a source-ordered 17-stage Python pipeline. It covers goal-directed engagement, online representation, local proposition extraction, questioning, attention allocation, candidate answer generation, learner explanation, elaboration, explanation checking and repair, teaching preparation, optional imagery, episodic binding, bounded plasticity, metacognitive monitoring, and closure.

The module deliberately does not implement broad prior-knowledge integration, spaced repetition, formal retrieval testing, interleaving, cognitive-load management, or motivation. Those remain separate source points.

## Package layout

- contracts.py: validated requests, segments, configuration, persistent learner state, and results.
- registry.py: canonical registry with P01-01 through P01-17 identifiers, source step numbers, inputs, outputs, equations, and biological cautions.
- functional.py: deterministic lexical/proposition operations, similarity, softmax, entropy, information-value proxies, explanation assessment, and episode-vector construction.
- biology.py: recurrent neural-population, frontoparietal attention, prefrontal control, hippocampal binding, and bounded plasticity surrogates.
- pipeline.py: the end-to-end processor and SQLite recording interface.
- annotations.py: entry point for AST registration in the shared ORE SQLite annotation catalogue.

## Mathematical operations implemented

The functional layer provides normalized competitive selection (softmax), Shannon entropy, cosine similarity, deterministic signed-hash feature vectors, coverage and evidence-support estimates, a weighted comprehension-quality proxy, prediction/uncertainty bookkeeping, and bounded updates.

The biological surrogate uses recurrent population integration:

    activity[t+1] = tanh(gain * input + W_recurrent @ activity[t] + bias - inhibition)

It uses attention weights to model graded competition and gain control, a goal-sensitive learning gate, and a hippocampal-style episode representation built from co-active feature vectors. Plasticity combines a centered Hebbian covariance term with a timing-sensitive potentiation/depression term. Connection weights are clipped to [-1, 1] to keep this model numerically bounded.

These formulas are candidate computational descriptions. They are not claims that the human brain explicitly computes softmax, cosine similarity, entropy, or these exact matrix equations.

## Important implementation safeguards

1. If the caller provides learner_explanation, the pipeline evaluates it as learner-generated text.
2. If no explanation is provided, the pipeline creates an explicitly labeled extractive scaffold from source material. It does not claim that the learner independently generated or understood it.
3. A scaffold alone cannot satisfy the final mastery/closure criterion.
4. Repair suggestions are linked to source passages. The system never fills an evidence gap with an invented fact.
5. Teaching output and imagery are optional.
6. Imagery output is symbolic scene/process data, not a rendered picture.
7. Brier calibration is computed only if independent proposition-accuracy labels and matching confidence estimates are supplied.
8. The algorithm stops at active-engagement episode closure; it does not schedule spaced repetitions or run a formal testing programme.
9. Biological modules are explicitly labeled computational surrogates. They are not biophysical simulations or neuron-level models.

## SQLite annotations and execution events

register_annotations(database_path) uses the repository's ore_logger.catalog.sync_repository to index every class and function in ore_active_engagement/ into the existing annotations table. The logger stores source path, qualified name, signature, docstring-derived annotation, line range, and source hash.

record_result_to_sqlite(result, database_path) registers the package definitions and stores 17 ordered operation events in the shared run/event tables. Each event retains the operation identifier, mathematical form, functional output, biological output, and evidence-level caution.

## Basic usage

    from ore_active_engagement import (
        ActiveEngagementProcessor,
        ActiveEngagementRequest,
        record_result_to_sqlite,
    )

    request = ActiveEngagementRequest(
        learner_id="learner-1",
        goal="understand selective transport across the cell membrane",
        material=(
            "The cell membrane forms a boundary around the cell. "
            "It regulates which substances enter and leave. "
            "Selective transport helps maintain stable internal conditions."
        ),
        learner_explanation=(
            "The cell membrane controls the movement of substances into and out "
            "of the cell, helping keep its internal environment stable."
        ),
        required_concepts=("membrane", "substances", "enter", "leave"),
        enable_teaching=True,
        enable_imagery=True,
    )

    processor = ActiveEngagementProcessor("learner-1")
    result = processor.process(request)
    print(result.output)
    record_result_to_sqlite(result, "ore_logs/annotations.sqlite3")

The source-to-code implementation is a transparent research prototype. Its text segmentation and vector features are intentionally simple and deterministic so that each operation can be inspected and tested.
