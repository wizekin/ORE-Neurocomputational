"""Process reward interfaces and verifiable reward collection (Section 5)."""
from .hacking import (  # noqa: F401
    HackingVerdict,
    IndependentTestSuite,
    detect_reward_hacking,
    split_visible_hidden,
)
from .process import (  # noqa: F401
    PRMModel,
    ReasoningStep,
    ReasoningTrace,
    StepKind,
    StepStatus,
)
from .rlvr import (  # noqa: F401
    RewardHackingError,
    RewardRecord,
    compute_rlvr_reward,
    replay_reward,
    reward_from_validators,
)
from .validators import (  # noqa: F401
    BUILTIN_VALIDATORS,
    DeterministicValidator,
    apply_validators,
    available_validators,
)

__all__ = (
    "BUILTIN_VALIDATORS",
    "HackingVerdict",
    "IndependentTestSuite",
    "PRMModel",
    "ReasoningStep",
    "ReasoningTrace",
    "RewardHackingError",
    "RewardRecord",
    "StepKind",
    "StepStatus",
    "apply_validators",
    "available_validators",
    "compute_rlvr_reward",
    "detect_reward_hacking",
    "replay_reward",
    "reward_from_validators",
    "split_visible_hidden",
)
