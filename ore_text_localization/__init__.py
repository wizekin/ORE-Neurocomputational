"""Raw text-image localization and its cooperative ORE visual-language pathway."""
from .annotations import register_visual_language_annotations
from .config import TextLocalizationConfig
from .contracts import (
    LocalizedGlyph,
    SegmentationHypothesis,
    TextImageRequest,
    TextLocalizationResult,
    TrainingGlyph,
)
from .localization import TextLocalizer, normalize_grayscale_image, otsu_threshold
from .pathway import (
    N9WorkingMemoryAdapter,
    VisualLanguagePathway,
    VisualLanguagePathwayResult,
)
from .registry import (
    LOCALIZATION_OPERATION_REGISTRY,
    PathwayOperation,
    build_visual_language_manifest,
    operation_registry_as_dicts,
)
from .segmentation import GlyphBox, SegmentationResult, segment_ink_mask
from .coordinator import (
    IMAGE_CONTRACT_ID,
    PATHWAY_MODULE_ID,
    PATHWAY_RESULT_CONTRACT_ID,
    VisualLanguageModuleAdapter,
    build_visual_language_task,
    coordinate_visual_language_input,
    create_text_image_artifact,
    register_visual_language_module,
)

__all__ = [
    "GlyphBox",
    "IMAGE_CONTRACT_ID",
    "LOCALIZATION_OPERATION_REGISTRY",
    "LocalizedGlyph",
    "N9WorkingMemoryAdapter",
    "PATHWAY_MODULE_ID",
    "PATHWAY_RESULT_CONTRACT_ID",
    "PathwayOperation",
    "SegmentationHypothesis",
    "SegmentationResult",
    "TextImageRequest",
    "TextLocalizationConfig",
    "TextLocalizationResult",
    "TextLocalizer",
    "TrainingGlyph",
    "VisualLanguageModuleAdapter",
    "VisualLanguagePathway",
    "VisualLanguagePathwayResult",
    "build_visual_language_manifest",
    "build_visual_language_task",
    "coordinate_visual_language_input",
    "create_text_image_artifact",
    "normalize_grayscale_image",
    "operation_registry_as_dicts",
    "otsu_threshold",
    "register_visual_language_annotations",
    "register_visual_language_module",
    "segment_ink_mask",
]
