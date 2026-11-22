# ORE Stage H — Implementation and Verification Report

**Version:** 0.6.0  
**Date:** 2026-11-13  
**Implemented scope:** H1–H3  
**Upstream contract:** persisted Stage G recognition result and manifest  
**Downstream contract:** versioned JSON interchange bundle; optional verified-feedback adapter for Stage I; integrity checks required by Stage J  
**Status:** Software implementation complete; scientific acceptance remains pending independent real-corpus evaluation.

## 1. Purpose and architectural position

Stage H is the serialization and provenance boundary between Stage G recognition and later ORE consumers. Stage G performs the bounded visual/orthographic inference; Stage H does not recognize text again, alter ranking, apply semantic correction, or update any model. It converts the persisted Stage G output into a stable, versioned interchange representation so later stages and independent programs do not need to depend on Python dataclass internals.

The implementation preserves the system boundaries established earlier:

- A provides the active ScriptProfile fingerprint and ORE error contracts.
- B/C provide visual patch and grapheme identity evidence already summarized by G.
- D provides positional, ordinal and pairwise-order evidence.
- E/F provide frozen orthographic patterns and seven-channel fusion evidence.
- G provides ranked grapheme sequences, abstention, and the exact parent/input lineage.
- I can consume Stage H-linked corrections only when a caller separately supplies verified graphemes and verification provenance.
- J requires Stage H's module, tests and report as part of the complete A–J software release.

## 2. H1 — Validate and serialize orthographic representation

1. Accept either a StageGResult object or a Stage G output directory.
2. Read both `recognition_result.json` and `recognition_manifest.json`; do not reconstruct missing provenance from in-memory assumptions.
3. Require agreement for Stage G ID, Stage F parent, input hash, checkpoint hash, script-profile fingerprint and recognition status.
4. Require the read-only recognition declarations and the uncalibrated-sequence-confidence declaration to remain explicit.
5. Validate candidate grapheme arrays, rendered forms, score finiteness, channel values, segmentation IDs and descending rank order.
6. Serialize the script-profile fingerprint, candidate state, character observations, order evidence and the unit conventions.

Output: `orthographic_representation.json`.

The `sequence` unit is a grapheme as defined by the active ScriptProfile, not a byte or an assumed single Unicode code point. The position unit is Stage G's logical patch index. The representation records omitted evidence sections when the compact-output configuration is selected.

## 3. H2 — Preserve ranked hypotheses and uncertainty

Output: `ranked_hypotheses.json` and `uncertainty.json`.

Each candidate preserves:

- explicit one-based rank;
- exact grapheme tuple and rendered string;
- Stage G beam mass and log score;
- visual log likelihood;
- the seven separately named weighted evidence-channel contributions (unless explicitly omitted by configuration);
- supporting segmentation-hypothesis IDs and best segmentation ID;
- an explicit uncalibrated confidence status.

The uncertainty artifact records top-candidate margin, entropy conditional on the returned candidate set, returned candidate mass, unreported mass, character entropy summaries, unknown mass, segmentation-hypothesis support and abstention reasons.

Important interpretation rules:

- Stage G sequence beam mass is not a calibrated probability that a candidate is correct.
- Top-K entropy is conditional on the returned candidate set; it is not the entropy of every possible string.
- Stage G currently does not export a normalized segmentation posterior. Stage H therefore records `NOT_EXPORTED_AS_DISTRIBUTION_BY_STAGE_G` instead of fabricating one.
- An `ABSTAIN` result remains an abstention after serialization; Stage H never promotes it to recognition success.

## 4. H3 — Reproducibility, lineage and integrity

Outputs: `provenance.json` and `stage_h_manifest.json`, plus `stage_h_manifest.sha256`.

The provenance records parent Stage G artifacts and their SHA-256 digests, Stage C/D/E/F/I lineage IDs, the frozen Stage F checkpoint digest, input fingerprint, script-profile fingerprint, configuration and fingerprint, Python/runtime context, and immutability assertions.

The three semantic data artifacts are written as canonical UTF-8 JSON with sorted keys, compact separators and non-finite numbers rejected. Their SHA-256 values are assembled into a bundle fingerprint. The provenance artifact is then written and pinned by the final Stage H manifest; the manifest is itself protected by a separate checksum sidecar.

`load_stage_h_bundle()` validates the sidecar, all four artifact hashes, schema identifiers and artifact/manifest ID agreement before returning the interchange bundle.

## 5. H-to-I feedback boundary

`make_stage_i_feedback_example()` is an opt-in bridge. It requires the caller to provide `graphemes`, `verification_status`, `source_id` and `verification_reference`. It does not select or copy the top Stage G candidate as a label. Instead, it creates a Stage I `IncrementalExample` tied to the Stage G recognition ID, Stage H ID and Stage H bundle fingerprint.

Stage I was extended compatibly:

- legacy examples without Stage H linkage retain their original example fingerprint format;
- Stage H linkage fields must be provided together;
- a Stage H-linked example must also retain its Stage G recognition ID;
- the SHA-256 field must be a lowercase 64-character digest;
- the Stage I overlay records the count and set of Stage H bundle fingerprints used.

Stage I still independently enforces its accepted verification classes and train/validation/test separation. A checksum proves artifact identity, not label truth. The caller's verification declaration remains an auditable trust boundary.

## 6. Public API

```python
from ore_stage_h import (
    StageHConfig,
    serialize_stage_h,
    load_stage_h_bundle,
    make_stage_i_feedback_example,
)

stage_h = serialize_stage_h(
    stage_g_result,  # Stage G must already have persisted both JSON artifacts
    config=StageHConfig(),
)

bundle = load_stage_h_bundle(stage_h.output_directory)
print(bundle["ranked_hypotheses.json"]["candidates"])

# Only after an independent correction/label check:
verified = make_stage_i_feedback_example(
    stage_h.output_directory,
    example_id="correction-001",
    graphemes=("c", "a", "b", "d"),
    split="train",
    verification_status="HUMAN_CONFIRMED",
    source_id="reviewed-corrections-v1",
    verification_reference="review-record-001",
)
```

A subsequent Stage I call must still include its required validation examples and satisfy all partition constraints.

## 7. Tests and verification

The dedicated Stage H suite contains 13 tests covering:

1. End-to-end serialization and bundle loading.
2. StageGResult-object and output-directory inputs.
3. Grapheme identity and all seven evidence channels.
4. Uncalibrated beam warning, returned mass and omitted mass.
5. Explicit absence of a segmentation posterior.
6. Compact evidence-output configuration.
7. Stage G source immutability.
8. Stage G manifest/result mismatch rejection.
9. Candidate grapheme/render mismatch rejection.
10. Stage H artifact tampering detection.
11. Stage H manifest-sidecar tampering detection.
12. Verified Stage I feedback adapter and lineage pinning.
13. Invalid configuration rejection.

The final package-wide regression suite passed 130/130 tests. Stage J accepted the complete A–J software package, verified the ZIP CRC and rejected duplicate archive member paths. The actual acceptance report and release manifest are shipped in the ZIP.

## 8. Limitations and future-facing contract

- Stage H does not add full-page text detection, text-region localization or automatic image segmentation.
- It does not infer semantic meaning, repair spelling, or use linguistic context to change the orthographic ranking.
- It does not calibrate sequence probabilities; that would require a separate validation design and held-out evidence.
- Segmentation probability distributions are not available from the current Stage G interface and are marked as unavailable.
- The interchange schema is versioned and intentionally JSON-based so future stages (including downstream evaluation, active-learning orchestration, multimodal input, or semantic processing) can consume it without mutating upstream models.
- Software integration tests do not establish reading accuracy, generalization across scripts/fonts, biological fidelity or AGI capability. Those remain separate scientific questions requiring rights-cleared real corpora, preregistered splits, baselines, multiple seeds and independent review.
