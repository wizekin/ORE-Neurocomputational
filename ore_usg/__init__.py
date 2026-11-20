"""Read-only adapters connecting versioned USG resources to ORE cognitive modules."""
from .contracts import (
    ORE_USG_BUNDLE_CONTRACT,
    ORE_USG_LEXICON_CONTRACT,
    USG_KNOWLEDGE_CONTRACT,
    USG_KNOWLEDGE_SCHEMA_VERSION,
    USG_LEXICAL_ADAPTER_VERSION,
    USGImportExclusion,
    USGLexiconImportReport,
    USGLexiconResource,
    USGSourceSummary,
    USGEvidenceSummary,
    USGContextSummary,
)
from .lexical import USGKnowledgeLexicalAdapter
from .bundle_io import (
    USGBundleLoadError,
    USGBundleSnapshot,
    discover_knowledge_bundle,
    load_knowledge_bundle,
)
from .policy import USGImportPolicy
from .structures import (
    USGKnowledgeRelation,
    USGPredicateFrame,
    USGSemanticClass,
    USGSemanticStructureAdapter,
    USGStructureImportReport,
    USGStructureResource,
)
from .semantic import (
    USGConceptAdapter,
    USGConceptMapping,
    USGSemanticImportReport,
    USGSemanticResource,
)
from .registration import (
    USG_LEXICAL_CAPABILITIES,
    USG_LEXICAL_MODULE_ID,
    USG_LEXICAL_MODULE_VERSION,
    build_usg_lexical_manifest,
    make_usg_bundle_artifact,
    register_usg_contracts,
    register_usg_lexical_module,
)
from .validation import (
    USGKnowledgeValidationError,
    require_valid_knowledge_bundle,
    validate_knowledge_bundle,
)

__all__ = [
    "ORE_USG_BUNDLE_CONTRACT",
    "ORE_USG_LEXICON_CONTRACT",
    "USG_KNOWLEDGE_CONTRACT",
    "USG_KNOWLEDGE_SCHEMA_VERSION",
    "USG_LEXICAL_ADAPTER_VERSION",
    "USGImportExclusion",
    "USGLexiconImportReport",
    "USGLexiconResource",
    "USGSourceSummary",
    "USGEvidenceSummary",
    "USGContextSummary",
    "USGKnowledgeLexicalAdapter",
    "USGImportPolicy",
    "USGSemanticStructureAdapter",
    "USGSemanticClass",
    "USGPredicateFrame",
    "USGKnowledgeRelation",
    "USGStructureImportReport",
    "USGStructureResource",
    "USGConceptAdapter",
    "USGConceptMapping",
    "USGSemanticImportReport",
    "USGSemanticResource",
    "USGBundleLoadError",
    "USGBundleSnapshot",
    "discover_knowledge_bundle",
    "load_knowledge_bundle",
    "USG_LEXICAL_CAPABILITIES",
    "USG_LEXICAL_MODULE_ID",
    "USG_LEXICAL_MODULE_VERSION",
    "build_usg_lexical_manifest",
    "make_usg_bundle_artifact",
    "register_usg_contracts",
    "register_usg_lexical_module",
    "USGKnowledgeValidationError",
    "require_valid_knowledge_bundle",
    "validate_knowledge_bundle",
]
