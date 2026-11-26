"""Optional participant coordination, aggregation and privacy controls (Section 5)."""
from .coordinator import (  # noqa: F401
    FederatedCoordinator,
    FederatedRound,
    ParticipantUpdate,
)
from .privacy import (  # noqa: F401
    FederatedPolicyError,
    ParticipantCredentials,
    PoisoningScreen,
    ThreatModel,
    protect_update,
    screen_poisoning,
    threat_model_summary,
)

__all__ = (
    "FederatedCoordinator",
    "FederatedPolicyError",
    "FederatedRound",
    "ParticipantCredentials",
    "ParticipantUpdate",
    "PoisoningScreen",
    "ThreatModel",
    "protect_update",
    "screen_poisoning",
    "threat_model_summary",
)
