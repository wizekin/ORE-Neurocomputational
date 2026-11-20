"""Point 24 feedback, clarification, and misconception correction."""
from .contracts import (
    ClarificationOption, CorrectionRequest, CorrectiveConfig, CorrectiveResult,
    CorrectiveState, FeedbackEvidence, KnowledgeHypothesis, RetrievalAttempt, TransferProbe,
)
from .pipeline import (
    Point24CorrectiveLearningProcessor, record_result_to_sqlite, register_annotations,
)
from .registry import OPERATION_REGISTRY, registry_as_dicts

__all__ = [
    "ClarificationOption", "CorrectionRequest", "CorrectiveConfig", "CorrectiveResult",
    "CorrectiveState", "FeedbackEvidence", "KnowledgeHypothesis", "RetrievalAttempt",
    "TransferProbe", "Point24CorrectiveLearningProcessor", "record_result_to_sqlite",
    "register_annotations", "OPERATION_REGISTRY", "registry_as_dicts",
]

from .recurrent_contracts import (
    ConflictDimension,
    ConflictReport,
    CorrectionRecord,
    FeedbackAction,
    FeedbackEpisode,
    RecurrentFeedbackConfig,
    RecurrentFeedbackRequest,
    RecurrentFeedbackResult,
)
from .feedback_scheduler import FeedbackActionScheduler
from .module_adapters import ModuleAdapterResult, RecurrentModuleAdapters
from .recurrent_controller import (
    RECURRENT_FEEDBACK_OPERATIONS,
    RecurrentFeedbackController,
    record_recurrent_result_to_sqlite,
    register_recurrent_annotations,
)

__all__ += [
    "ConflictDimension", "ConflictReport", "CorrectionRecord", "FeedbackAction",
    "FeedbackEpisode", "RecurrentFeedbackConfig", "RecurrentFeedbackRequest",
    "RecurrentFeedbackResult", "FeedbackActionScheduler", "ModuleAdapterResult",
    "RecurrentModuleAdapters", "RECURRENT_FEEDBACK_OPERATIONS",
    "RecurrentFeedbackController", "record_recurrent_result_to_sqlite",
    "register_recurrent_annotations",
]
