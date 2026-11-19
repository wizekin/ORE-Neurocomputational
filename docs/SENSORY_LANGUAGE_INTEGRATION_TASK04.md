# Task 4 — Connect sensory processing to language and semantic understanding

## Scope

The \`ore_text_localization\` package connects a raw text image to ORE's existing visual and language modules. It deliberately does not replace N1–N9, the lexical-access engine, syntax, semantic composition, discourse processing, or working memory.

The specialist processing path is:

\`TextImageRequest → TextLocalizer → N9/N1–N4 → N6 lexical access → syntax → N8 semantic composition → N7 discourse → working memory\`

The completed Task 4 pathway is itself registered as one typed cognitive module in Task 3's \`CognitiveTaskCoordinator\`. Task 3 plans and executes the registered module, validates its output artifact and provenance, and publishes artifact references and module status into the shared workspace. The specialist processors remain owned by their existing packages and are invoked through N9 adapters.

## What is new

- Raw image normalization, adaptive polarity handling and Otsu thresholding.
- Horizontal/vertical projection-based line and word localization, with caller-declared reading direction.
- Candidate glyph boxes and fixed-size, aspect-preserving grayscale crops compatible with \`N9Patch\`.
- Unsplit alternatives for low-ink valley cuts, with probabilities that sum to one.
- Explicit N9-to-\`WorkingMemoryRequest\` conversion that preserves tokens, syntax, propositions and unresolved dependencies.
- Optional supervised N3 fitting using caller-labelled \`TrainingGlyph\` samples.
- A versioned module manifest and a mathematical operation registry.
- Typed \`ContractArtifact\` adapters and contract definitions for Task 3's module registry.
- SQLite annotation registration through the existing \`ore_logger.annotations\` table.

## Mathematical operations

### Image normalization

For RGB samples, luminance is approximated as

\[
Y = 0.2126R + 0.7152G + 0.0722B,
\]

then scaled/clipped into \([0,1]\). Integer pixels are scaled by the dtype maximum. Foreground polarity is estimated from the border median so light text on a dark background can be canonicalized before N1 processing.

### Threshold selection

The Otsu threshold \(t^*\) maximizes between-class variance:

\[
t^* = \arg\max_t \; w_0(t)w_1(t)(\mu_0(t)-\mu_1(t))^2.
\]

This is a global thresholding heuristic, not a learned text detector.

### Line, word and glyph localization

For a binary ink mask \(B\), row and column projections are

\[
p_r = \sum_c B_{r,c}, \qquad q_c = \sum_r B_{r,c}.
\]

Contiguous active runs propose text lines and glyph components. A word boundary is proposed when the whitespace gap \(g_j\) exceeds

\[
g_j > \max(g_{\min}, \operatorname{round}(r_w h_{\text{line}})).
\]

For wide components, a smoothed ink projection is inspected for candidate valleys. The relative valley depth is

\[
d(c)=1-\frac{q_{\text{smooth}}(c)}
{\min(\max_{u<c}q_{\text{smooth}}(u),\max_{u>c}q_{\text{smooth}}(u))}.
\]

A cut is proposed only when its depth and minimum-width conditions are satisfied. Each cut keeps an unsplit alternative; the implementation does not label either boundary choice as certainly correct.

### Crop normalization

Each image crop is resized while preserving its aspect ratio and centered in a fixed \(32 \times 32\) grayscale canvas. The image patch and its original-page \((x_0,y_0,x_1,y_1)\) coordinates are retained separately.

### Sensory and language feedback

After the crops reach N9, the existing N3/N4 path can combine visual evidence and contextual predictions. The intended precision-weighted form is

\[
Q_i(g)=\operatorname{softmax}_g\left[
\pi_{s,i}\log(P_v(g\mid x_i)+\epsilon)
+\pi_{p,i}\log(P_t(g)+\epsilon)
\right].
\]

A bounded recurrent update can be expressed as

\[
Q_i^{(t+1)}=(1-\rho)Q_i^{(t)}+\rho\widehat Q_i^{(t+1)},
\qquad 0<\rho\leq1.
\]

N9 retains visual probabilities, predictive priors, posteriors and disagreement traces as separate fields; the new localizer does not overwrite raw sensory evidence. The equations describe the computational design, not a claim that neurons literally execute these software operations.

## Minimal image-to-N9 API

\`\`\`python
import numpy as np
from ore_text_localization import TextImageRequest, TextLocalizer

request = TextImageRequest(
    sequence_id="page-001",
    image=np.asarray(page_pixels),
    script_profile={
        "language": "en",
        "script": "Latn",
        "layout": "horizontal",
        "reading_direction": "ltr",
    },
)
localized = TextLocalizer().process(request)
n9_request = localized.to_n9_request(task="read")
\`\`\`

## Registering and running through Task 3's coordinator

Register the Task 4 manifest, exact-version input/output contracts and executable backend once in the coordinator's \`ModuleRegistry\`:

\`\`\`python
from ore_contracts import ContractCatalog, CognitiveTaskCoordinator, ModuleRegistry
from ore_text_localization import (
    VisualLanguagePathway,
    register_visual_language_module,
    coordinate_visual_language_input,
)

catalog = ContractCatalog()
registry = ModuleRegistry(catalog)

pathway = VisualLanguagePathway.from_processors(
    lexical_access=lexical_processor,         # caller-supplied lexicon
    syntax=syntax_processor,
    semantic_composition=semantic_processor, # caller-supplied concepts and frames
    discourse=discourse_processor,
    n3=trained_n3_processor,                 # fitted using labelled N1/N2 examples
)
register_visual_language_module(catalog, registry, pathway)

coordinator = CognitiveTaskCoordinator(registry, catalog)
task_result = coordinate_visual_language_input(coordinator, request)
\`\`\`

\`build_visual_language_task(request)\` can be used separately when the caller wants to inspect or enrich the \`CognitiveTask\` before submitting it. The helper constructs the versioned \`ore.text_image_request@1.0.0\` input artifact and requests the \`language.visual_pathway\` capability. The registered backend returns an \`ore.visual_language_pathway_result@1.0.0\` artifact with the input artifact fingerprint in its parent lineage.

The Task 3 coordinator executes this module in its dependency-aware plan, validates the module identity, output contract, run ID and artifact lineage, then publishes module status and artifact references to the shared workspace. The internal N9 pathway still delegates to the existing N1–N4, N6, syntax, N8, N7 and working-memory implementations.

The result status is \`COMPLETED\` only when the fitted N3 model, lexical/syntax/semantic/discourse adapters, semantic evidence and working-memory handoff all report usable outputs. Otherwise it is \`PARTIAL\` or \`UNAVAILABLE\`; missing inventories are not silently synthesized. The manifest declares these specialist resources as optional dependencies because the present backend invokes them in-process through N9; they are not falsely represented as independently scheduled Task 3 backends.

## Annotation database

Calling \`register_visual_language_annotations()\` indexes every class, function, nested function and method introduced in \`ore_text_localization\` through the existing \`ore_logger.sync_repository\` API. The target is \`ore_logs/ore_logger.sqlite3\` unless \`database_path\` or \`ORE_LOGGER_DB\` is provided. Definitions are stored in the shared \`ore_logger.annotations\` SQLite catalogue; this does not create a second annotation database. Existing N1–N9, language, working-memory and Task 3 coordinator packages retain their own annotation registrations and are not redundantly rewritten.

\`VisualLanguagePathway\` enables registration by default. Set \`register_annotations=False\` only when a caller explicitly manages annotation registration elsewhere.

## Scientific and engineering limits

- Projection segmentation is a constrained printed-text proposal. It is not general OCR and is not validated for handwriting, Arabic joining/ligatures, severe skew, multi-column pages or complex layouts.
- N3 requires caller-labelled examples covering the intended grapheme set. A fitted classifier is not evidence of general recognition accuracy.
- Lexical entries, morphology, concept vectors and predicate frames must be supplied by their owning modules.
- A successful software handoff does not establish recognition accuracy, human-like comprehension or biological validity.
- Direct callers may still use the optional callback or workspace publisher; normal Task 3 integration uses the registered module adapter and typed artifacts.
