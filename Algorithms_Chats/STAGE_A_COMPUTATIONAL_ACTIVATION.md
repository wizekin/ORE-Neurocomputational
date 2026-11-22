# Stage A — Computational Activation

**Pipeline version:** 1.0.0  
**Package baseline:** ORE Workbench 0.8.0  
**Scope:** executable deterministic baseline over caller-pre-segmented glyph patches.  
**Scientific status:** software-tested research prototype; independent real-corpus evaluation remains pending.

## 1. What changed

`initialize_ore(...)` now attaches `context.stage_a_pipeline`, an orchestrated implementation of all seven Stage A modules. Initialization and computational execution are separate traceable events. Merely initializing a run does not claim that glyphs were processed; each actual sample emits one execution event per module and a final pipeline-completed event.

The baseline is intentionally not a duplicate of later trained stages:

- Stage A visual encoding is a deterministic handcrafted grid descriptor.
- Stage A identity is a transparent nearest-prototype baseline fitted only from explicitly labeled reference patches.
- Stage C remains the trainable visual encoder and supervised/self-supervised representation learner.
- Stage D remains the learned absolute/relative/distributed position backend.
- Stage A orthographic memory is a separate run-local diagnostic memory; it does not modify Stage E or Stage I persisted memory.

## 2. Module algorithms and mathematical definitions

### A5 — Visual Encoder

**Input:** rectangular grayscale patch `X ∈ [0,1]^(H×W)`, `patch_id`.  
**Output:** fixed-dimensional feature vector `z ∈ R^d`, mean intensity, and contrast.

1. Bilinearly resample the image to a grid containing `d` sample locations.
2. Flatten the sampled values into `x ∈ R^d`.
3. Center the vector: `x_c = x - mean(x)·1`.
4. Normalize: `z = x_c / max(||x_c||_2, ε)`.

Uniform patches therefore produce the zero vector instead of NaN. This feature is spatial contrast, not a learned visual representation.

### A6 — Character Identity

**Fit input:** explicitly labeled reference patches `(X_i, y_i)`.  
**Inference input:** encoded feature vector `z`.  
**Output:** ranked class probabilities, predicted grapheme or profile unknown token, confidence, and decision status.

For each grapheme `c`, calculate the mean reference vector and L2-normalize it to produce prototype `μ_c`. Then:

`logit(c|z) = cosine(z, μ_c) / τ_id`

`P(c|z) = softmax(logit(c|z) / T_cal)`

`τ_id = max(0.05, 0.15 × config.score_temperature)` and `T_cal` is the calibration temperature (1.0 until calibrated). If the highest posterior is below `abstention_threshold`, return the profile's `<UNK>` token. Labeled references must belong to the active script inventory after profile normalization. No class mapping is invented when references are absent.

### A7 — Absolute Position

**Input:** the sample's patch IDs and optional bounding boxes.  
**Output:** normalized physical/logical positions and Gaussian slot activations.

When bounding boxes are provided, use the center coordinate on the profile's configured position axis. Normalize the coordinate across the sample to `[0,1]`. For RTL scripts, reverse horizontal coordinates to obtain logical reading position. When boxes are absent, use input order as the logical order. For slot center `c_j = j/(K-1)`:

`a_j(p) = exp(-0.5 × ((p-c_j)/σ_pos)^2)`

Normalize activations so `Σ_j a_j(p)=1`.

### A8 — Relative Position

**Input:** absolute logical positions and reading order.  
**Output:** one record for every ordered pair `(i,j)` with `i` before `j`.

`Δ_ij = p_j - p_i`, and `distance_ij = |Δ_ij|`.

The sign encodes the relative displacement; `relation` labels the result as later, earlier, or same-position.

### A9 — Position Overlap

**Input:** normalized logical positions.  
**Output:** a soft pairwise similarity in `[0,1]`.

`O_ij = exp(-((p_i-p_j)^2)/(2 σ_overlap^2))`.

This Gaussian kernel is an engineering proxy for overlapping positional codes. It is not asserted to be the exact receptive-field function of a brain region.

### A10 — Orthographic Memory

**Input:** high-confidence grapheme predictions explicitly accepted for memory update.  
**Output:** lifetime token counts, exponentially decayed recent counts, and contiguous grapheme n-gram counts with JSON-encoded token-array keys (so multi-codepoint graphemes cannot collide).

Before each update, `R_t(k) = λ R_(t-1)(k)`. Then each accepted token increments lifetime and recent counts. Contiguous n-grams of lengths `2..min(max_chunk_length,n)` are counted. Unknown and below-`confidence_threshold` predictions are excluded. Memory is run-local and written to `stage_a_orthographic_memory.json`; it is not Stage E's pattern store or Stage I's checksummed overlay.

### A11 — Calibration

**Fit input:** held-out labeled calibration examples encoded by the fitted identity model.  
**Output:** temperature `T_cal`, example count, and pre/post negative log-likelihood.

A deterministic 121-point log-spaced search over `[0.05,20]` selects the temperature minimizing mean NLL. Calibration rejects exact pixel-matrix overlap with the identity-reference set and duplicate calibration matrices so the caller cannot accidentally calibrate on an exact copy of a prototype patch. With no held-out examples, calibration remains explicitly `default_uncalibrated`; probabilities are not described as empirically calibrated.

## 3. Public Python input/output contract

```python
from ore_stage_a import OREConfig, initialize_ore, latin_basic_profile
from ore_stage_a_pipeline import GlyphReference, GlyphPatch, StageASample

context = initialize_ore(OREConfig(), latin_basic_profile(), "./ore_runs")
pipeline = context.stage_a_pipeline

pipeline.fit([
    GlyphReference("a", pixels_a, reference_id="ref-a"),
    GlyphReference("b", pixels_b, reference_id="ref-b"),
])

result = pipeline.process(StageASample("sample-001", [
    GlyphPatch("patch-001", pixels_a, bbox=(0, 0, 20, 24)),
    GlyphPatch("patch-002", pixels_b, bbox=(22, 0, 42, 24)),
]))
print(result.recognized_text)
```

Pixel matrices are sequences of equal-length rows with finite float/int values from 0.0 (black) to 1.0 (white). Bounding boxes, when supplied, use `(left, top, right, bottom)` with positive width and height. Each matrix represents one already-segmented grapheme. Automatic page detection and grapheme segmentation are outside this API.

`pipeline.process(sample, update_memory=True)` returns a `StageAResult` containing encoded features, predictions, absolute positions, relative pairs, overlap pairs, calibration report, memory update count, and a stable result SHA-256. Set `update_memory=False` for read-only processing.

## 4. Workbench Chat JSON contract

Attach `examples/workbench/stage_a_request.example.json` and enter:

```text
run stage A on attached glyphs
```

Top-level fields:

- `script_profile`: `latin_basic` or `arabic_basic`.
- `seed`: optional non-negative integer.
- `references`: optional array of `{reference_id, label, pixels}`.
- `calibration_examples`: optional array of `{label, pixels}`; requires references.
- `samples`: non-empty array of `{sample_id, glyphs}`.
- Each glyph: `{patch_id, pixels, bbox?}`.
- `update_memory`: optional boolean, defaults to true.

The chat operation returns the actual run ID, module statuses, per-sample output, trace path, and saved artifacts. It does not claim recognition when references are absent; the identity module emits `<UNK>` while the other six computational modules still execute.

## 5. Evidence artifacts and trace events

Initialization writes `stage_a_pipeline.json` and appends `STAGE_A_PIPELINE_READY` after `ORE_INITIALIZED`.

If references are fitted, the pipeline writes `stage_a_identity_model.json` and records `STAGE_A_IDENTITY_PROTOTYPES_FIT`; optional calibration records `STAGE_A_CALIBRATION_FIT`.

For every processed sample it writes `stage_a_result_<sample_id>.json` and, when memory is changed, `stage_a_orthographic_memory.json`. The trace contains:

- `STAGE_A_PIPELINE_STARTED`
- seven `STAGE_A_MODULE_EXECUTED` events (one per named module)
- `STAGE_A_PIPELINE_COMPLETED`

Each module event identifies its algorithm, input/output counts, and relevant parameters. The final output hash excludes run ID so identical computational outputs across fresh runs have a matching fingerprint.

## 6. Status vocabulary

`implemented_*` means code exists and was attached to the run; it does not mean the module was trained or scientifically validated. Character identity remains `implemented_initialized_untrained_no_reference_prototypes` until labeled references are fitted. Calibration remains `implemented_default_uncalibrated_temperature` until held-out examples are fitted. The overall ORE model remains `INITIALIZED_UNTRAINED` until its later training lifecycle changes that state.

## 7. Validation performed

The dedicated Stage A initialization and pipeline tests cover input validation, deterministic feature vectors, untrained abstention, labeled prototype fitting, LTR/RTL geometry, relative and overlap channels, high-confidence-only memory updates, calibration persistence, execution traces, artifact persistence, and repeatability across fresh runs.

The full package regression suite must still be run on the target Windows environment after installation. Software tests demonstrate implementation and reproducibility of the declared baseline; they do not establish real-world OCR accuracy, biological fidelity, or scientific acceptance. Independent evaluation on rights-cleared real script/font/image corpora remains pending.
