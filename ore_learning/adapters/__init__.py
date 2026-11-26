"""Text, vision, audio, video, spatial and framework adapters (Section 5)."""
from .base import (  # noqa: F401
    AdapterOutputError,
    AdapterRegistry,
    ExecutionContext,
    SpecialistAdapter,
    validate_payload_against_schema,
)
from .fusion import (  # noqa: F401
    FusedConcept,
    FusionResult,
    ModalityObservation,
    SemanticMapper,
    UNKNOWN_USG_CONCEPT,
    fuse_observations,
    map_to_semantics,
)
from .mock import (  # noqa: F401
    BrokenAdapter,
    CountingAdapter,
    MockCaptionAdapter,
    MockRegressionAdapter,
    MockSpeechAdapter,
    MockSpatialAdapter,
    MockTextSentimentAdapter,
    MockVisionAdapter,
    build_mock_registry,
)
from .modality import (  # noqa: F401
    ModalityPayload,
    ModalityValidationError,
    NormalizedModalityInput,
    ObservationGateway,
)

__all__ = (
    "AdapterOutputError",
    "AdapterRegistry",
    "BrokenAdapter",
    "CountingAdapter",
    "ExecutionContext",
    "FusedConcept",
    "FusionResult",
    "MockCaptionAdapter",
    "MockRegressionAdapter",
    "MockSpeechAdapter",
    "MockSpatialAdapter",
    "MockTextSentimentAdapter",
    "MockVisionAdapter",
    "ModalityObservation",
    "ModalityPayload",
    "ModalityValidationError",
    "NormalizedModalityInput",
    "ObservationGateway",
    "SemanticMapper",
    "SpecialistAdapter",
    "UNKNOWN_USG_CONCEPT",
    "build_mock_registry",
    "fuse_observations",
    "map_to_semantics",
    "validate_payload_against_schema",
)
