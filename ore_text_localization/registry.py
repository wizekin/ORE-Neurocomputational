"""Versioned pathway manifest and mathematical operation registry."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from ore_contracts.module_manifest import (
    InputPort,
    ModuleDependency,
    ModuleIdentity,
    ModuleManifest,
    OutputPort,
)


@dataclass(frozen=True, slots=True)
class PathwayOperation:
    """Describe one explicit computational operation and the mathematical rule it applies."""

    operation_id: str
    name: str
    objective: str
    mathematical_operation: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible operation record for traces and documentation."""
        return asdict(self)


LOCALIZATION_OPERATION_REGISTRY = (
    PathwayOperation(
        "TLP-001", "Normalize image luminance",
        "Convert grayscale or RGB pixels to bounded, comparable intensity values.",
        "Y = 0.2126 R + 0.7152 G + 0.0722 B; scale and clip to [0, 1].",
    ),
    PathwayOperation(
        "TLP-002", "Estimate foreground threshold",
        "Separate likely ink from background without using a learned grapheme identity.",
        "t* = argmax_t [w0(t) w1(t) (mu0(t) - mu1(t))^2] (Otsu between-class variance).",
    ),
    PathwayOperation(
        "TLP-003", "Localize text lines",
        "Project binary foreground counts onto the row or column axis and group contiguous runs.",
        "p_r = sum_c B[r,c]; line runs = maximal intervals where p_r >= m.",
    ),
    PathwayOperation(
        "TLP-004", "Locate word boundaries",
        "Group nearby glyph candidates unless the measured whitespace exceeds a line-height threshold.",
        "split_word iff gap_j > max(g_min, round(r_word * line_height)).",
    ),
    PathwayOperation(
        "TLP-005", "Propose glyph boundaries",
        "Find sufficiently deep local minima in the ink projection of wide connected candidates.",
        "depth(c) = 1 - p_smooth(c) / min(max_left, max_right); retain if depth >= theta.",
    ),
    PathwayOperation(
        "TLP-006", "Normalize glyph crops",
        "Preserve aspect ratio while interpolating each crop into the fixed N9 input canvas.",
        "s = min(W_available / w, H_available / h); center resized crop in a fixed canvas.",
    ),
    PathwayOperation(
        "TLP-007", "Preserve segmentation uncertainty",
        "Retain the primary split and bounded unsplit alternatives with explicit normalized weights.",
        "P(H_0)=0.8; P(H_j)=0.2/K for K alternatives; no candidate is labelled as a grapheme.",
    ),
    PathwayOperation(
        "TLP-008", "Combine sensory and prior evidence",
        "Delegate to N9/N4 and retain the visual and predictive channels separately.",
        "Q_i(g)=softmax[pi_s,i log(P_v(g|x_i)+eps)+pi_p,i log(P_t(g)+eps)].",
    ),
    PathwayOperation(
        "TLP-009", "Apply bounded recurrent feedback",
        "Allow contextual/lexical feedback to revise the posterior without overwriting raw visual evidence.",
        "Q^(t+1)=(1-rho) Q^t + rho Q_hat^(t+1), with 0 < rho <= 1.",
    ),
)

if tuple(item.operation_id for item in LOCALIZATION_OPERATION_REGISTRY) != tuple(
    f"TLP-{index:03d}" for index in range(1, 10)
):
    raise RuntimeError("TLP operation IDs must be consecutive from TLP-001 to TLP-009")


def operation_registry_as_dicts() -> list[dict[str, Any]]:
    """Return all localization operations and mathematical rules for callers and tests."""
    return [item.to_dict() for item in LOCALIZATION_OPERATION_REGISTRY]


def build_visual_language_manifest() -> ModuleManifest:
    """Describe Task 4 using the real typed manifest consumed by Task 3's module registry."""
    return ModuleManifest(
        identity=ModuleIdentity(
            module_id="language.visual_pathway",
            module_version="1.0.0",
            module_name="Visual localization to language and semantic understanding",
            package_path="ore_text_localization",
            biological_reference=(
                "Computational coordination of visual form, lexical/language and working-memory "
                "functions; not a literal anatomical route or measured neural connectome."
            ),
        ),
        responsibilities=(
            "Normalize a raw page image and propose text-line, word and glyph boundaries.",
            "Preserve spatial order, word spans, segmentation alternatives and source coordinates.",
            "Hand normalized glyph observations to the established ORE N9 visual-language pathway.",
            "Pass supported lexical, syntactic, semantic and discourse outputs into working memory.",
            "Return a typed result artifact to the Task 3 CognitiveTaskCoordinator.",
        ),
        input_ports=(
            InputPort(
                port_id="raw_text_image",
                contract_id="ore.text_image_request",
                accepted_versions=("1.0.0",),
                expected_data_type="ore_text_localization.contracts.TextImageRequest",
                validation_requirements=("finite_pixels", "valid_image_shape", "declared_or_default_script_profile"),
            ),
        ),
        output_ports=(
            OutputPort(
                port_id="pathway_result",
                contract_id="ore.visual_language_pathway_result",
                contract_version="1.0.0",
                output_data_type="ore_text_localization.pathway.VisualLanguagePathwayResult",
                description="Ordered glyphs, lexical alternatives, syntax, semantic evidence and working-memory state.",
            ),
        ),
        capabilities=(
            "visual.text_localization",
            "visual.glyph_segmentation",
            "language.grapheme_recognition",
            "language.lexical_access",
            "language.syntax_parsing",
            "language.semantic_composition",
            "language.discourse_integration",
            "cognition.working_memory_handoff",
            "language.visual_pathway",
        ),
        dependencies=(
            ModuleDependency("visual.early_processing", required=False, purpose="N1/N2 sensory feature extraction is invoked in-process by N9"),
            ModuleDependency("visual.orthographic_recognition", required=False, purpose="N3 may be fitted with caller-supplied labelled data"),
            ModuleDependency("visual.attention_precision", required=False, purpose="N4 attention and precision gating is invoked in-process by N9"),
            ModuleDependency("language.lexical_access", required=False, purpose="N6 lexical candidate adapter may be injected"),
            ModuleDependency("language.syntax_parsing", required=False, purpose="syntax processor may be injected"),
            ModuleDependency("language.semantic_composition", required=False, purpose="N8 concept and proposition processor may be injected"),
            ModuleDependency("language.discourse_integration", required=False, purpose="N7 discourse processor may be injected"),
            ModuleDependency("cognition.working_memory", required=False, purpose="working-memory handoff uses the repository processor"),
        ),
        execution_modes=("FUNCTIONAL", "HYBRID", "INFER", "VALIDATE"),
        limitations=(
            "Text localization is a constrained projection-based proposal, not a general OCR model.",
            "N3 requires caller-supplied labelled training examples before grapheme recognition is available.",
            "Lexical and semantic inventories must be supplied; missing facts are not invented.",
            "Task 3's coordinator schedules this pathway as one typed cognitive module; its specialist computations remain owned by their existing packages.",
            "Rate-coded and neural-reference models remain computational abstractions, not measurements.",
        ),
    )


__all__ = [
    "LOCALIZATION_OPERATION_REGISTRY",
    "PathwayOperation",
    "build_visual_language_manifest",
    "operation_registry_as_dicts",
]
